use bevy::{
    asset::RenderAssetUsages,
    prelude::*,
    render::mesh::{Indices, PrimitiveTopology},
};
use std::f32::consts::TAU;

const WINDOW_SIZE: u32 = 640;
/// Horizontal pixel budget for the grid — leaves 80 px on each side for
/// arrow-slide animations and future edge chrome.
const GRID_AVAILABLE_X: f32 = 480.0;
/// Vertical pixel budget — matches the horizontal margin for now; reduce this
/// when a top HUD (score, controls) is added.
const GRID_AVAILABLE_Y: f32 = 480.0;
const MIN_GRID_DIM: i32 = 10;
const MAX_GRID_DIM: i32 = 40;
/// Dot radius as a fraction of `cell_size`. Keeps dots proportional to arrows
/// so small cells on large grids still look right.
const DOT_RADIUS_FRAC: f32 = 0.06;
const BACKGROUND_COLOR: Color = Color::srgb(0.08, 0.08, 0.12);
const DOT_COLOR: Color = Color::srgb(0.35, 0.35, 0.45);

// ── Game states ──────────────────────────────────────────────────────────────

#[derive(States, Debug, Clone, PartialEq, Eq, Hash, Default)]
enum GameState {
    /// Title / splash screen.
    #[default]
    Intro,
    /// Active puzzle play.
    Playing,
    /// Shown after solving a puzzle; displays progress before moving on.
    PuzzleComplete,
}

// ── Level ────────────────────────────────────────────────────────────────────

/// Tracks which level (0-indexed) the player is on.
#[derive(Resource, Default)]
struct LevelIndex(u32);

/// Tracks the highest level number the player has completed (1-based display).
/// Zero means no level has been completed yet.
#[derive(Resource, Default)]
struct HighestLevel(u32);

/// Returns the `(cols, rows)` grid dimensions for a given level index.
/// Deterministic but pseudo-random: derived from a hash of the level number.
fn level_grid_size(level: u32) -> (i32, i32) {
    let h = hash_u32(level);
    let range = (MAX_GRID_DIM - MIN_GRID_DIM + 1) as u32;
    let cols = MIN_GRID_DIM + (h % range) as i32;
    let rows = MIN_GRID_DIM + ((h >> 8) % range) as i32;
    (cols, rows)
}

/// Returns an integer cell spacing (in pixels) that fits the grid within the
/// available viewport. Each axis is constrained independently so non-square
/// grids use the space efficiently.
fn cell_size_for_grid(cols: i32, rows: i32) -> f32 {
    let from_x = GRID_AVAILABLE_X / (cols - 1) as f32;
    let from_y = GRID_AVAILABLE_Y / (rows - 1) as f32;
    from_x.min(from_y).floor().max(1.0)
}

/// Bijective integer hash (Wang / MurmurHash3 finalizer).
fn hash_u32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x45d9f3b);
    x ^= x >> 16;
    x
}

// ── Arrow ────────────────────────────────────────────────────────────────────

/// A single arrow in the puzzle.
/// Arrows are independent ECS entities tagged with `PlayfieldEntity`, so they
/// are despawned automatically when leaving `GameState::Playing`.
#[derive(Component)]
struct Arrow {
    /// Path of the arrow as a sequence of grid coordinates `(col, row)`.
    vertices: Vec<IVec2>,
    /// Creation-order index assigned by the level generation algorithm.
    order: usize,
    /// oklch hue in degrees [0, 360) — used to recolour on hover/animation.
    hue: f32,
}

/// Sliding animation state attached to every arrow entity.
#[derive(Component, Default)]
struct ArrowSlide {
    /// Current displacement in world units. Zero when at rest.
    offset: f32,
    /// Direction of sliding in world space (normalised). Set when the level
    /// loads; not meaningful while `offset` is zero.
    direction: Vec2,
}

// ── Grid occupancy map ──────────────────────────────────────────────────────

/// Maps every grid cell to the [`Entity`] of the arrow that occupies it.
///
/// Populated by `setup_arrows`; removed when leaving `GameState::Playing`.
/// Every cell on every axis-aligned segment of each arrow path is marked, so
/// a single cell look-up is enough to identify which arrow was clicked.
#[derive(Resource)]
struct GridMap {
    cols: i32,
    rows: i32,
    /// Row-major flat storage: `cells[row * cols + col]`.
    cells: Vec<Option<Entity>>,
}

impl GridMap {
    fn new(cols: i32, rows: i32) -> Self {
        Self {
            cols,
            rows,
            cells: vec![None; (cols * rows) as usize],
        }
    }

    /// Returns the entity at `(col, row)`, or `None` if empty / out-of-bounds.
    fn get(&self, col: i32, row: i32) -> Option<Entity> {
        if col < 0 || row < 0 || col >= self.cols || row >= self.rows {
            return None;
        }
        self.cells[(row * self.cols + col) as usize]
    }

    /// Writes `entity` into `(col, row)`. Silently ignores out-of-bounds writes.
    fn set(&mut self, col: i32, row: i32, entity: Option<Entity>) {
        if col >= 0 && row >= 0 && col < self.cols && row < self.rows {
            self.cells[(row * self.cols + col) as usize] = entity;
        }
    }

    /// Marks every cell on the axis-aligned segment `a → b` with `entity`.
    fn mark_segment(&mut self, a: IVec2, b: IVec2, entity: Entity) {
        if a.x == b.x {
            let (r0, r1) = (a.y.min(b.y), a.y.max(b.y));
            for row in r0..=r1 {
                self.set(a.x, row, Some(entity));
            }
        } else {
            let (c0, c1) = (a.x.min(b.x), a.x.max(b.x));
            for col in c0..=c1 {
                self.set(col, a.y, Some(entity));
            }
        }
    }
}

/// Tracks which arrow entity (if any) the pointer is currently hovering.
#[derive(Resource, Default)]
struct HoveredArrow(Option<Entity>);

/// Tracks the grid cell `(col, row)` currently under the pointer, if any.
#[derive(Resource, Default)]
struct HoveredCell(Option<IVec2>);

/// Marks a grid-dot entity with its logical grid position.
#[derive(Component)]
struct GridDot {
    col: i32,
    row: i32,
}

// ── Arrow mesh building ───────────────────────────────────────────────────────────

/// Arrow body half-width as a fraction of `cell_size`.
const ARROW_HALF_WIDTH_FRAC: f32 = 0.14;
/// Arrowhead length (base to tip) as a fraction of `cell_size`.
const ARROWHEAD_LENGTH_FRAC: f32 = 0.55;
/// Arrowhead base half-width as a fraction of `cell_size`.
const ARROWHEAD_HALF_WIDTH_FRAC: f32 = 0.36;
/// Triangles used per rounded corner arc and end cap.
const ROUND_SEGMENTS: usize = 5;
/// Arrow slide speed in world units (pixels) per second.
const SLIDE_SPEED: f32 = 500.0;
/// oklch lightness for an idle, un-hovered arrow.
const ARROW_LIGHTNESS_NORMAL: f32 = 0.62;
/// oklch lightness while the pointer is hovering over an arrow.
const ARROW_LIGHTNESS_HOVERED: f32 = 0.80;
/// oklch lightness while an arrow is animating off the board.
const ARROW_LIGHTNESS_ANIMATING: f32 = 0.92;
/// oklch chroma — moderate saturation gives a clear but not garish look.
const ARROW_CHROMA: f32 = 0.13;

/// Accumulates 2-D triangle-list geometry.
#[derive(Default)]
struct MeshBuilder {
    positions: Vec<Vec2>,
    indices: Vec<u32>,
}

impl MeshBuilder {
    /// Push a vertex and return its index.
    fn push(&mut self, pos: Vec2) -> u32 {
        let idx = self.positions.len() as u32;
        self.positions.push(pos);
        idx
    }

    /// Add one CCW-wound triangle.
    fn tri(&mut self, a: u32, b: u32, c: u32) {
        self.indices.extend([a, b, c]);
    }

    /// Add a quad as two CCW triangles.
    /// Pass vertices in CCW order: top-left, bottom-left, bottom-right, top-right.
    fn quad(&mut self, tl: u32, bl: u32, br: u32, tr: u32) {
        self.tri(tl, bl, br);
        self.tri(tl, br, tr);
    }

    /// Fan of triangles from `center_idx` that fill the arc of radius `hw`
    /// around `center` sweeping from `start` to `end`.
    /// `ccw = true` sweeps counter-clockwise (positive angle direction).
    fn arc_fan(
        &mut self,
        center_idx: u32,
        center: Vec2,
        start: Vec2,
        end: Vec2,
        hw: f32,
        ccw: bool,
    ) {
        let start_angle = (start - center).to_angle();
        let end_angle = (end - center).to_angle();
        let span = if ccw {
            let d = end_angle - start_angle;
            if d <= 0.0 { d + TAU } else { d }
        } else {
            let d = end_angle - start_angle;
            if d >= 0.0 { d - TAU } else { d }
        };
        let mut prev_idx = self.push(start);
        for i in 1..=ROUND_SEGMENTS {
            let t = i as f32 / ROUND_SEGMENTS as f32;
            let angle = start_angle + span * t;
            let curr = center + Vec2::from_angle(angle) * hw;
            let curr_idx = self.push(curr);
            if ccw {
                self.tri(center_idx, prev_idx, curr_idx);
            } else {
                self.tri(center_idx, curr_idx, prev_idx);
            }
            prev_idx = curr_idx;
        }
    }

    /// Consume the builder and produce a renderable [`Mesh`].
    fn build(self) -> Mesh {
        let n = self.positions.len();
        let positions: Vec<[f32; 3]> = self.positions.iter().map(|p| [p.x, p.y, 0.0]).collect();
        let normals = vec![[0.0_f32, 0.0, 1.0]; n];
        let uvs = vec![[0.0_f32, 0.0]; n];
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
        mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
        mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
        mesh.insert_indices(Indices::U32(self.indices));
        mesh
    }
}

/// Build the 2-D [`Mesh`] for an arrow in its current animation state.
///
/// Implements a "snake" / lightcycle animation: both ends of the arrow travel
/// along the path at the same speed so the body length stays constant.
/// - The **head** extends past the last vertex in the direction of the final
///   segment by `slide.offset` world units.
/// - The **tail** is clipped: the first `slide.offset` world units of the
///   original path are removed.
/// When `slide.offset == 0` the arrow is drawn in its resting position.
fn build_arrow_mesh(
    vertices: &[IVec2],
    cell_size: f32,
    grid_origin: Vec2,
    slide: &ArrowSlide,
) -> Mesh {
    assert!(vertices.len() >= 2, "arrow needs at least two vertices");

    let hw = cell_size * ARROW_HALF_WIDTH_FRAC;
    let head_len = cell_size * ARROWHEAD_LENGTH_FRAC;
    let head_hw = cell_size * ARROWHEAD_HALF_WIDTH_FRAC;

    // Convert grid coords → world-space base path.
    let base: Vec<Vec2> = vertices
        .iter()
        .map(|v| grid_origin + v.as_vec2() * cell_size)
        .collect();
    let nb = base.len();

    // Per-segment (direction, length) for the base path.
    let base_segs: Vec<(Vec2, f32)> = (0..nb - 1)
        .map(|i| {
            let d = base[i + 1] - base[i];
            let len = d.length().max(f32::EPSILON);
            (d / len, len)
        })
        .collect();
    let total_len: f32 = base_segs.iter().map(|s| s.1).sum();
    let last_dir = base_segs[nb - 2].0;

    let t = slide.offset.max(0.0);

    // ── Extended head: past the final vertex by t ─────────────────────────────
    let head = base[nb - 1] + last_dir * t;

    // ── Trimmed tail: advance t along the path ────────────────────────────────
    // Returns (segment_index, position_on_that_segment).
    // segment_index == base_segs.len() means the tail is past all original verts.
    let (tail_seg, tail_pos) = if t >= total_len {
        (base_segs.len(), base[nb - 1] + last_dir * (t - total_len))
    } else {
        let mut rem = t;
        let mut result = (0usize, base[0]);
        for i in 0..base_segs.len() {
            if rem < base_segs[i].1 {
                result = (i, base[i] + base_segs[i].0 * rem);
                break;
            }
            rem -= base_segs[i].1;
            // Reached the end of segment i; tail is at base[i+1].
            result = (i + 1, base[i + 1]);
        }
        result
    };

    // ── Animated path: tail_pos → intermediate verts → head ──────────────────
    // Include original vertices between tail_seg+1 and nb-2 (nb-1 is replaced
    // by head so the arrowhead always follows the extended tip).
    let mut wps: Vec<Vec2> = vec![tail_pos];
    for i in (tail_seg + 1)..(nb - 1) {
        wps.push(base[i]);
    }
    wps.push(head);

    let n = wps.len();

    // Per-segment direction and left-perpendicular (CCW rotation of direction).
    let dirs: Vec<Vec2> = (0..n - 1)
        .map(|i| (wps[i + 1] - wps[i]).normalize())
        .collect();
    let perps: Vec<Vec2> = dirs.iter().map(|d| Vec2::new(-d.y, d.x)).collect();

    let mut mb = MeshBuilder::default();

    // ── Tail cap: semicircle facing backward from the first vertex ────────────
    {
        let v = wps[0];
        let p = perps[0];
        let c = mb.push(v);
        // CCW arc from +perp to −perp sweeps through the backward half-circle.
        mb.arc_fan(c, v, v + p * hw, v - p * hw, hw, true);
    }

    // ── Segment quads and rounded corner joins ────────────────────────────────
    for i in 0..n - 1 {
        let a = wps[i];
        // Shorten the final segment to leave room for the arrowhead.
        let b = if i == n - 2 {
            wps[n - 1] - dirs[i] * head_len
        } else {
            wps[i + 1]
        };
        let p = perps[i];

        let tl = mb.push(a + p * hw);
        let bl = mb.push(a - p * hw);
        let br = mb.push(b - p * hw);
        let tr = mb.push(b + p * hw);
        mb.quad(tl, bl, br, tr);

        // Rounded join between this segment and the next.
        if i < n - 2 {
            let v = wps[i + 1];
            let pn = perps[i + 1];
            // Cross product sign determines turn direction.
            let cross = dirs[i].x * dirs[i + 1].y - dirs[i].y * dirs[i + 1].x;
            let vc = mb.push(v);
            if cross > 1e-4 {
                // Left turn (CCW): outer gap is on the −perp (right) side.
                mb.arc_fan(vc, v, v - p * hw, v - pn * hw, hw, true);
            } else if cross < -1e-4 {
                // Right turn (CW): outer gap is on the +perp (left) side.
                mb.arc_fan(vc, v, v + p * hw, v + pn * hw, hw, false);
            }
            // Straight continuation: the quads already meet flush.
        }
    }

    // ── Arrowhead: triangle at the last vertex ────────────────────────────────
    {
        // Push the tip slightly past the grid point so the dot sits inside.
        // base stays at wps[n-1] - head_len to match the end of the segment quad.
        let tip = wps[n - 1] + dirs[n - 2] * (head_len * 0.25);
        let p = perps[n - 2];
        let base = wps[n - 1] - dirs[n - 2] * head_len;
        let tip_idx = mb.push(tip);
        let left_idx = mb.push(base + p * head_hw); // +perp = left side
        let right_idx = mb.push(base - p * head_hw); // −perp = right side
        // CCW: tip → left → right
        mb.tri(tip_idx, left_idx, right_idx);
    }

    mb.build()
}

// ── Level generation ──────────────────────────────────────────────────────────

/// Specification for one arrow to be spawned by `setup_arrows`. Produced by
/// the generator; consumed when the arrow's ECS entity is created.
struct ArrowSpec {
    /// Path vertices from tail end to arrowhead, in grid coordinates.
    vertices: Vec<IVec2>,
    /// Random hue in degrees [0, 360).
    hue: f32,
}

/// Maximum distance (in cells) from a board-edge cell to its arrowhead.
const MAX_HEAD_DEPTH: i32 = 4;
/// Maximum number of axis-aligned segments per arrow tail.
const MAX_SEGMENTS: i32 = 4;
/// Inclusive bounds on a single tail segment's length (in cells).
const MIN_SEG_LEN: i32 = 1;
const MAX_SEG_LEN: i32 = 5;

/// Tiny seeded RNG built on `hash_u32`. Deterministic per seed and produces
/// reasonable-quality output for level generation — not for cryptography.
struct Rng(u32);

impl Rng {
    fn new(seed: u32) -> Self {
        Self(hash_u32(seed.wrapping_add(1)))
    }
    fn next_u32(&mut self) -> u32 {
        self.0 = hash_u32(self.0.wrapping_add(0x9E3779B9));
        self.0
    }
    /// Returns a uniform integer in `[lo, hi]` (both inclusive).
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        let span = (hi - lo + 1) as u32;
        lo + (self.next_u32() % span) as i32
    }
    fn bool(&mut self) -> bool {
        self.next_u32() & 1 == 0
    }
    /// Returns a uniform `f32` in `[0, 1)`.
    fn f32_unit(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
    /// In-place Fisher–Yates shuffle.
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u32() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

#[inline]
fn cell_index(p: IVec2, cols: i32) -> usize {
    (p.y * cols + p.x) as usize
}

#[inline]
fn in_bounds(p: IVec2, cols: i32, rows: i32) -> bool {
    p.x >= 0 && p.y >= 0 && p.x < cols && p.y < rows
}

/// Rotate a cardinal direction 90° CCW (player's left when facing `d`).
#[inline]
fn turn_left(d: IVec2) -> IVec2 {
    IVec2::new(-d.y, d.x)
}

/// Rotate a cardinal direction 90° CW (player's right when facing `d`).
#[inline]
fn turn_right(d: IVec2) -> IVec2 {
    IVec2::new(d.y, -d.x)
}

/// Mark every cell on the axis-aligned segment `a → b` as occupied.
fn mark_segment_occupied(occupied: &mut [bool], cols: i32, a: IVec2, b: IVec2) {
    if a.x == b.x {
        let (r0, r1) = (a.y.min(b.y), a.y.max(b.y));
        for r in r0..=r1 {
            occupied[cell_index(IVec2::new(a.x, r), cols)] = true;
        }
    } else {
        let (c0, c1) = (a.x.min(b.x), a.x.max(b.x));
        for c in c0..=c1 {
            occupied[cell_index(IVec2::new(c, a.y), cols)] = true;
        }
    }
}

/// Generates a level deterministically from `seed`. Returns arrows in
/// **solve order**: `result[0]` is the arrow the player must click first,
/// `result[n-1]` is the last to be cleared.
///
/// Algorithm (reverse-order placement): the arrow placed *first* in the
/// generator is the *last* one the player solves, since by the time the
/// player gets to it every earlier-solved arrow has already left the board.
/// Each iteration picks an unused outlet on the board edge, looks for a
/// valid arrowhead position with a clear corridor to that edge, then
/// random-walks a tail backward from the arrowhead. Only body cells are
/// recorded as occupied — corridor cells stay free so later-placed (=
/// earlier-solved) arrows may cross them.
fn generate_level(cols: i32, rows: i32, seed: u32) -> Vec<ArrowSpec> {
    let mut rng = Rng::new(seed);
    let mut occupied = vec![false; (cols * rows) as usize];
    let mut placed: Vec<ArrowSpec> = Vec::new();

    // (edge_cell, inward_direction) outlets — one per boundary cell per edge
    // it touches. Corner cells contribute two outlets (one per adjacent edge).
    let mut outlets: Vec<(IVec2, IVec2)> = Vec::new();
    for c in 0..cols {
        outlets.push((IVec2::new(c, 0), IVec2::Y)); // bottom edge → inward = up
        outlets.push((IVec2::new(c, rows - 1), IVec2::NEG_Y)); // top edge → inward = down
    }
    for r in 0..rows {
        outlets.push((IVec2::new(0, r), IVec2::X)); // left edge → inward = right
        outlets.push((IVec2::new(cols - 1, r), IVec2::NEG_X)); // right edge → inward = left
    }
    rng.shuffle(&mut outlets);

    for (edge_cell, inward) in outlets {
        // Arrowhead's slide direction is outward (toward the edge cell).
        let head_dir = -inward;
        let mut depths: Vec<i32> = (1..=MAX_HEAD_DEPTH).collect();
        rng.shuffle(&mut depths);

        for depth in depths {
            let arrowhead = edge_cell + inward * depth;
            if !in_bounds(arrowhead, cols, rows) {
                continue;
            }
            if occupied[cell_index(arrowhead, cols)] {
                continue;
            }
            // Check the corridor (arrowhead+head_dir … edge cell) is clear.
            let mut corridor_clear = true;
            let mut step = arrowhead + head_dir;
            while in_bounds(step, cols, rows) {
                if occupied[cell_index(step, cols)] {
                    corridor_clear = false;
                    break;
                }
                step += head_dir;
            }
            if !corridor_clear {
                continue;
            }
            // Build a per-attempt corridor mask so the tail walk can avoid it.
            let mut corridor = vec![false; (cols * rows) as usize];
            let mut step = arrowhead + head_dir;
            while in_bounds(step, cols, rows) {
                corridor[cell_index(step, cols)] = true;
                step += head_dir;
            }

            if let Some(verts) = try_place_arrow(
                cols, rows, &occupied, &corridor, arrowhead, head_dir, &mut rng,
            ) {
                for win in verts.windows(2) {
                    mark_segment_occupied(&mut occupied, cols, win[0], win[1]);
                }
                let hue = rng.f32_unit() * 360.0;
                placed.push(ArrowSpec {
                    vertices: verts,
                    hue,
                });
                break;
            }
        }
    }

    // Solve order is reverse of placement order.
    placed.reverse();
    placed
}

/// Random-walks a tail backward from `arrowhead` and assembles the arrow's
/// vertex list. Returns `Some([tail_end, …corners…, arrowhead])` on success
/// (at least one tail cell beyond the arrowhead), or `None` if the first
/// step is already blocked.
fn try_place_arrow(
    cols: i32,
    rows: i32,
    occupied: &[bool],
    corridor: &[bool],
    arrowhead: IVec2,
    head_dir: IVec2,
    rng: &mut Rng,
) -> Option<Vec<IVec2>> {
    // Self-avoidance mask: the tail must not revisit its own cells.
    let mut visited = vec![false; (cols * rows) as usize];
    visited[cell_index(arrowhead, cols)] = true;

    // End of each completed segment, in order moving outward from the arrowhead.
    let mut segment_ends: Vec<IVec2> = Vec::new();
    let mut cur = arrowhead;
    let mut dir = -head_dir;

    let num_segments = rng.range(1, MAX_SEGMENTS);
    for _ in 0..num_segments {
        let target = rng.range(MIN_SEG_LEN, MAX_SEG_LEN);
        let mut steps = 0;
        for _ in 0..target {
            let next = cur + dir;
            if !in_bounds(next, cols, rows) {
                break;
            }
            let i = cell_index(next, cols);
            if occupied[i] || corridor[i] || visited[i] {
                break;
            }
            visited[i] = true;
            cur = next;
            steps += 1;
        }
        if steps == 0 {
            break;
        }
        segment_ends.push(cur);
        if steps < target {
            break; // segment cut short — accept what we have, don't continue
        }
        dir = if rng.bool() {
            turn_left(dir)
        } else {
            turn_right(dir)
        };
    }

    if segment_ends.is_empty() {
        return None;
    }

    // Reverse so vertices read tail-end → corners → arrowhead.
    let mut vertices = Vec::with_capacity(segment_ends.len() + 1);
    for p in segment_ends.iter().rev() {
        vertices.push(*p);
    }
    vertices.push(arrowhead);
    Some(vertices)
}

// ── Marker components (used for state-scoped entity cleanup) ─────────────────

#[derive(Component)]
struct IntroScreen;

#[derive(Component)]
struct PlayfieldEntity;

#[derive(Component)]
struct PuzzleCompleteScreen;

// ── App ───────────────────────────────────────────────────────────────────────

fn main() {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "Vector Emancipation".into(),
                resolution: (640_u32, 640_u32).into(),
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(BACKGROUND_COLOR))
        .add_plugins(MeshPickingPlugin)
        .insert_resource(MeshPickingSettings {
            require_markers: true,
            ..default()
        })
        .init_state::<GameState>()
        .init_resource::<LevelIndex>()
        .init_resource::<HighestLevel>()
        .init_resource::<HoveredArrow>()
        .init_resource::<HoveredCell>()
        .add_systems(Startup, setup_camera)
        // Intro
        .add_systems(OnEnter(GameState::Intro), setup_intro)
        .add_systems(OnExit(GameState::Intro), teardown::<IntroScreen>)
        .add_systems(Update, advance_intro.run_if(in_state(GameState::Intro)))
        // Playing
        .add_systems(OnEnter(GameState::Playing), (setup_playfield, setup_arrows))
        .add_systems(
            OnExit(GameState::Playing),
            (teardown::<PlayfieldEntity>, remove_grid_map),
        )
        .add_systems(
            Update,
            (
                animate_arrows,
                rebuild_arrow_meshes,
                update_arrow_colors,
                check_level_complete,
            )
                .chain()
                .run_if(in_state(GameState::Playing)),
        )
        .add_systems(
            Update,
            update_dot_scales.run_if(in_state(GameState::Playing)),
        )
        // Puzzle complete
        .add_systems(OnEnter(GameState::PuzzleComplete), setup_puzzle_complete)
        .add_systems(
            OnExit(GameState::PuzzleComplete),
            teardown::<PuzzleCompleteScreen>,
        )
        .add_systems(
            Update,
            advance_puzzle_complete.run_if(in_state(GameState::PuzzleComplete)),
        )
        .run();
}

// ── Shared systems ────────────────────────────────────────────────────────────

fn setup_camera(mut commands: Commands) {
    commands.spawn((Camera2d, MeshPickingCamera));
}

/// Despawns every entity tagged with `T` (and their descendants).
fn teardown<T: Component>(mut commands: Commands, query: Query<Entity, With<T>>) {
    for entity in &query {
        commands.entity(entity).despawn();
    }
}

/// Removes the [`GridMap`] resource and resets [`HoveredArrow`] when leaving
/// `GameState::Playing`.
fn remove_grid_map(mut commands: Commands) {
    commands.remove_resource::<GridMap>();
    commands.insert_resource(HoveredArrow::default());
    commands.insert_resource(HoveredCell::default());
}

// ── Intro ─────────────────────────────────────────────────────────────────────

fn setup_intro(mut commands: Commands) {
    commands
        .spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                row_gap: Val::Px(20.0),
                ..default()
            },
            IntroScreen,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Vector Emancipation"),
                TextFont {
                    font_size: FontSize::Px(52.0),
                    ..default()
                },
                TextColor(Color::WHITE),
            ));
            parent.spawn((
                Text::new(
                    "Long ago, in the Cartesian Plane, a few\n\
                    courageous vectors sought to free themselves\n\
                    from the domination of the Origin (0, 0).",
                ),
                TextFont {
                    font_size: FontSize::Px(17.0),
                    ..default()
                },
                TextColor(Color::srgb(0.65, 0.65, 0.78)),
                TextLayout {
                    justify: Justify::Center,
                    ..default()
                },
            ));
            parent.spawn((
                Text::new("Press any key to begin"),
                TextFont {
                    font_size: FontSize::Px(22.0),
                    ..default()
                },
                TextColor(Color::srgb(0.55, 0.55, 0.70)),
            ));
        });
}

fn advance_intro(
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    if keyboard.get_just_pressed().next().is_some() || mouse.get_just_pressed().next().is_some() {
        next_state.set(GameState::Playing);
    }
}

// ── Playing ───────────────────────────────────────────────────────────────────

fn setup_playfield(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    level: Res<LevelIndex>,
) {
    let (cols, rows) = level_grid_size(level.0);
    let cell_size = cell_size_for_grid(cols, rows);

    let dot_radius = (cell_size * DOT_RADIUS_FRAC).max(1.0);
    let dot_mesh = meshes.add(Circle::new(dot_radius));
    let dot_material = materials.add(ColorMaterial::from(DOT_COLOR));

    // Round offsets to whole pixels so every dot lands on a pixel boundary.
    let offset_x = (-(cols - 1) as f32 * cell_size / 2.0).round();
    let offset_y = (-(rows - 1) as f32 * cell_size / 2.0).round();

    for row in 0..rows {
        for col in 0..cols {
            let x = offset_x + col as f32 * cell_size;
            let y = offset_y + row as f32 * cell_size;
            commands.spawn((
                Mesh2d(dot_mesh.clone()),
                MeshMaterial2d(dot_material.clone()),
                Transform::from_xyz(x, y, 0.0),
                GridDot { col, row },
                PlayfieldEntity,
            ));
        }
    }

    // Invisible rectangle that covers the entire grid (+half-cell border).
    // Placed at z = 2 so it is always the topmost hit target; all interaction
    // goes through grid coordinates rather than individual entity picking.
    let hit_w = cols as f32 * cell_size;
    let hit_h = rows as f32 * cell_size;
    let hit_mesh = meshes.add(Rectangle::new(hit_w, hit_h));
    let transparent = materials.add(ColorMaterial::from(Color::srgba(0.0, 0.0, 0.0, 0.0)));
    commands
        .spawn((
            Mesh2d(hit_mesh),
            MeshMaterial2d(transparent),
            Transform::from_xyz(0.0, 0.0, 2.0),
            Pickable::default(),
            PlayfieldEntity,
        ))
        .observe(on_grid_click)
        .observe(on_grid_hover)
        .observe(on_grid_leave);
}

/// Observer: pointer moved over the grid — update [`HoveredArrow`] and [`HoveredCell`].
fn on_grid_hover(
    trigger: On<Pointer<Move>>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    level: Res<LevelIndex>,
    grid_map: Res<GridMap>,
    mut hovered: ResMut<HoveredArrow>,
    mut hovered_cell: ResMut<HoveredCell>,
) {
    let Ok((camera, cam_transform)) = camera_q.single() else {
        return;
    };
    let screen_pos = trigger.event().pointer_location.position;
    let Ok(world_pos) = camera.viewport_to_world_2d(cam_transform, screen_pos) else {
        return;
    };

    let (cols, rows) = level_grid_size(level.0);
    let cell_size = cell_size_for_grid(cols, rows);
    let grid_origin = Vec2::new(
        (-(cols - 1) as f32 * cell_size / 2.0).round(),
        (-(rows - 1) as f32 * cell_size / 2.0).round(),
    );

    let col = ((world_pos.x - grid_origin.x) / cell_size).round() as i32;
    let row = ((world_pos.y - grid_origin.y) / cell_size).round() as i32;
    let col = col.clamp(0, cols - 1);
    let row = row.clamp(0, rows - 1);

    let entity = grid_map.get(col, row);
    if hovered.0 != entity {
        hovered.0 = entity;
    }
    let cell = IVec2::new(col, row);
    if hovered_cell.0 != Some(cell) {
        hovered_cell.0 = Some(cell);
    }
}

/// Observer: pointer left the grid hit-area — clear [`HoveredArrow`] and [`HoveredCell`].
fn on_grid_leave(
    _trigger: On<Pointer<Out>>,
    mut hovered: ResMut<HoveredArrow>,
    mut hovered_cell: ResMut<HoveredCell>,
) {
    hovered.0 = None;
    hovered_cell.0 = None;
}

/// Observer fired when the player clicks anywhere on the grid hit-area.
/// Converts the viewport-space pointer position to `(col, row)` and starts
/// the arrow occupying that cell sliding (if it is currently idle).
fn on_grid_click(
    trigger: On<Pointer<Click>>,
    camera_q: Query<(&Camera, &GlobalTransform)>,
    level: Res<LevelIndex>,
    grid_map: Res<GridMap>,
    mut arrow_q: Query<(&Arrow, &mut ArrowSlide)>,
) {
    let Ok((camera, cam_transform)) = camera_q.single() else {
        return;
    };
    let screen_pos = trigger.event().pointer_location.position;
    let Ok(world_pos) = camera.viewport_to_world_2d(cam_transform, screen_pos) else {
        return;
    };

    let (cols, rows) = level_grid_size(level.0);
    let cell_size = cell_size_for_grid(cols, rows);
    let grid_origin = Vec2::new(
        (-(cols - 1) as f32 * cell_size / 2.0).round(),
        (-(rows - 1) as f32 * cell_size / 2.0).round(),
    );

    let col = ((world_pos.x - grid_origin.x) / cell_size).round() as i32;
    let row = ((world_pos.y - grid_origin.y) / cell_size).round() as i32;
    let col = col.clamp(0, cols - 1);
    let row = row.clamp(0, rows - 1);
    info!("Grid click at ({col}, {row})");

    let Some(entity) = grid_map.get(col, row) else {
        return; // empty cell
    };

    let Ok((arrow, mut slide)) = arrow_q.get_mut(entity) else {
        return;
    };
    if slide.offset > 0.0 {
        return; // already animating
    }
    let n = arrow.vertices.len();
    let last = (arrow.vertices[n - 1] - arrow.vertices[n - 2])
        .as_vec2()
        .normalize();
    slide.direction = last;
    slide.offset = 1.0;
}

fn setup_arrows(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    level: Res<LevelIndex>,
) {
    let (cols, rows) = level_grid_size(level.0);
    let cell_size = cell_size_for_grid(cols, rows);
    let grid_origin = Vec2::new(
        (-(cols - 1) as f32 * cell_size / 2.0).round(),
        (-(rows - 1) as f32 * cell_size / 2.0).round(),
    );

    let mut grid_map = GridMap::new(cols, rows);

    // Spawns one arrow, fills its cells in the grid map, and returns the entity.
    let mut spawn = |verts: Vec<IVec2>, order: usize, hue: f32, grid_map: &mut GridMap| -> Entity {
        let slide = ArrowSlide::default();
        let mesh = meshes.add(build_arrow_mesh(&verts, cell_size, grid_origin, &slide));
        let color = Color::oklch(ARROW_LIGHTNESS_NORMAL, ARROW_CHROMA, hue);
        let material = materials.add(ColorMaterial::from(color));
        let entity = commands
            .spawn((
                Arrow {
                    vertices: verts.clone(),
                    order,
                    hue,
                },
                slide,
                Mesh2d(mesh),
                MeshMaterial2d(material),
                Transform::from_xyz(0.0, 0.0, 1.0),
                PlayfieldEntity,
            ))
            .id();
        for seg in verts.windows(2) {
            grid_map.mark_segment(seg[0], seg[1], entity);
        }
        entity
    };

    for (order, spec) in generate_level(cols, rows, level.0).into_iter().enumerate() {
        spawn(spec.vertices, order, spec.hue, &mut grid_map);
    }

    commands.insert_resource(grid_map);
}

/// Advances each animating arrow and despawns it once it has slid off-screen.
fn animate_arrows(
    mut commands: Commands,
    time: Res<Time>,
    level: Res<LevelIndex>,
    mut query: Query<(Entity, &Arrow, &mut ArrowSlide)>,
) {
    let (cols, rows) = level_grid_size(level.0);
    let cell_size = cell_size_for_grid(cols, rows);
    for (entity, arrow, mut slide) in &mut query {
        if slide.offset <= 0.0 {
            continue;
        }
        slide.offset += SLIDE_SPEED * time.delta_secs();
        // Total path length in world units.
        let total_len: f32 = arrow
            .vertices
            .windows(2)
            .map(|w| ((w[1] - w[0]).as_vec2() * cell_size).length())
            .sum();
        // Despawn once the tail is well past the screen boundary.
        if slide.offset > total_len + WINDOW_SIZE as f32 {
            commands.entity(entity).despawn();
        }
    }
}

/// Rebuilds the mesh of every arrow whose [`ArrowSlide`] changed this frame.
fn rebuild_arrow_meshes(
    mut meshes: ResMut<Assets<Mesh>>,
    level: Res<LevelIndex>,
    query: Query<(&Arrow, &ArrowSlide, &Mesh2d), Changed<ArrowSlide>>,
) {
    let (cols, rows) = level_grid_size(level.0);
    let cell_size = cell_size_for_grid(cols, rows);
    let grid_origin = Vec2::new(
        (-(cols - 1) as f32 * cell_size / 2.0).round(),
        (-(rows - 1) as f32 * cell_size / 2.0).round(),
    );
    for (arrow, slide, mesh2d) in &query {
        if let Some(mut mesh) = meshes.get_mut(&mesh2d.0) {
            *mesh = build_arrow_mesh(&arrow.vertices, cell_size, grid_origin, slide);
        }
    }
}

/// Transitions to [`GameState::PuzzleComplete`] once every arrow has slid
/// off the playfield and been despawned.
fn check_level_complete(
    arrow_q: Query<(), With<Arrow>>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    if arrow_q.is_empty() {
        next_state.set(GameState::PuzzleComplete);
    }
}

/// Recolours every arrow each frame based on hover and animation state.
///
/// Three lightness tiers (all same chroma + hue):
/// - animating  → brightest
/// - hovered    → medium bright
/// - idle       → normal
fn update_arrow_colors(
    hovered: Res<HoveredArrow>,
    arrow_q: Query<(Entity, &Arrow, &ArrowSlide, &MeshMaterial2d<ColorMaterial>)>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    for (entity, arrow, slide, mat) in &arrow_q {
        let lightness = if slide.offset > 0.0 {
            ARROW_LIGHTNESS_ANIMATING
        } else if hovered.0 == Some(entity) {
            ARROW_LIGHTNESS_HOVERED
        } else {
            ARROW_LIGHTNESS_NORMAL
        };
        if let Some(mut material) = materials.get_mut(&mat.0) {
            material.color = Color::oklch(lightness, ARROW_CHROMA, arrow.hue);
        }
    }
}

/// Scales each grid dot to 150% when the pointer is over its cell, 100% otherwise.
/// Uses an early-out on [`HoveredCell`] change detection to avoid unnecessary work.
fn update_dot_scales(hovered_cell: Res<HoveredCell>, mut dot_q: Query<(&GridDot, &mut Transform)>) {
    if !hovered_cell.is_changed() {
        return;
    }
    for (dot, mut transform) in &mut dot_q {
        let target = if hovered_cell.0 == Some(IVec2::new(dot.col, dot.row)) {
            1.5
        } else {
            1.0
        };
        if transform.scale.x != target {
            transform.scale = Vec3::splat(target);
        }
    }
}

// ── Puzzle complete ───────────────────────────────────────────────────────────

fn setup_puzzle_complete(
    mut commands: Commands,
    level: Res<LevelIndex>,
    mut highest: ResMut<HighestLevel>,
) {
    // Record highest completed level (level.0 is 0-indexed, so +1 for display).
    highest.0 = highest.0.max(level.0 + 1);
    let highest_text = format!("Highest level: {}", highest.0);

    commands
        .spawn((
            Node {
                width: Val::Percent(100.0),
                height: Val::Percent(100.0),
                flex_direction: FlexDirection::Column,
                justify_content: JustifyContent::Center,
                align_items: AlignItems::Center,
                row_gap: Val::Px(20.0),
                ..default()
            },
            PuzzleCompleteScreen,
        ))
        .with_children(|parent| {
            parent.spawn((
                Text::new("Puzzle Complete!"),
                TextFont {
                    font_size: FontSize::Px(52.0),
                    ..default()
                },
                TextColor(Color::WHITE),
            ));
            parent.spawn((
                Text::new(highest_text),
                TextFont {
                    font_size: FontSize::Px(24.0),
                    ..default()
                },
                TextColor(Color::srgb(0.65, 0.80, 0.65)),
            ));
            parent.spawn((
                Text::new("Press any key to continue"),
                TextFont {
                    font_size: FontSize::Px(22.0),
                    ..default()
                },
                TextColor(Color::srgb(0.55, 0.55, 0.70)),
            ));
        });
}

/// Advances to the next level when the player presses a key or clicks.
fn advance_puzzle_complete(
    keyboard: Res<ButtonInput<KeyCode>>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut level: ResMut<LevelIndex>,
    mut next_state: ResMut<NextState<GameState>>,
) {
    if keyboard.get_just_pressed().next().is_some() || mouse.get_just_pressed().next().is_some() {
        level.0 += 1;
        next_state.set(GameState::Playing);
    }
}
