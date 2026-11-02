import json
import math
import warnings
from datetime import datetime
from pyproj import CRS, Transformer
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
import xml.etree.ElementTree as ET
from xml.dom import minidom


# Lane layout table, identical to determine_lane_pairs() in the Lanelet2 converter.
# Key: (number of boundary ways on the road, set of boundary way_ids)
# Value: list of (right_way, left_way, lane_no) per lane.
#   - The travel direction of a lane is the direction in which right_way is on the
#     right and left_way is on the left.
#   - |lane_no| is the lane position counted from the reference line (way 200/202/500/700).
#   - The centerline way_id of a lane is f"{min(right_way, left_way)}12".
LANE_LAYOUTS = {
    # 1-LANE SCENARIOS
    (2, frozenset({100, 500})): [(100, 500, -1)],
    (2, frozenset({300, 700})): [(300, 700, 1)],

    # 2-LANE SCENARIOS
    (3, frozenset({100, 200, 300})): [(100, 200, -1), (300, 200, 1)],
    (3, frozenset({100, 400, 500})): [(100, 400, -2), (400, 500, -1)],
    (3, frozenset({300, 800, 700})): [(300, 800, 2), (800, 700, 1)],

    # 3-LANE SCENARIOS
    (4, frozenset({100, 101, 202, 300})): [(100, 101, -2), (101, 202, -1), (300, 202, 1)],
    (4, frozenset({100, 200, 201, 300})): [(100, 200, -1), (201, 200, 1), (300, 201, 2)],
    (4, frozenset({100, 400, 401, 500})): [(100, 400, -3), (400, 401, -2), (401, 500, -1)],
    (4, frozenset({300, 800, 801, 700})): [(300, 800, 3), (800, 801, 2), (801, 700, 1)],

    # 4-LANE SCENARIOS
    (5, frozenset({100, 101, 200, 201, 300})):
        [(100, 101, -2), (101, 200, -1), (201, 200, 1), (300, 201, 2)],
    (5, frozenset({100, 400, 401, 402, 500})):
        [(100, 400, -4), (400, 401, -3), (401, 402, -2), (402, 500, -1)],
    (5, frozenset({300, 800, 801, 802, 700})):
        [(300, 800, 4), (800, 801, 3), (801, 802, 2), (802, 700, 1)],

    # 5-LANE SCENARIOS
    (6, frozenset({100, 101, 102, 200, 201, 300})):
        [(100, 101, -3), (101, 102, -2), (102, 200, -1), (201, 200, 1), (300, 201, 2)],
    (6, frozenset({100, 101, 202, 203, 204, 300})):
        [(100, 101, -2), (101, 202, -1), (203, 202, 1), (204, 203, 2), (300, 204, 3)],
    (6, frozenset({100, 400, 401, 402, 403, 500})):
        [(100, 400, -5), (400, 401, -4), (401, 402, -3), (402, 403, -2), (403, 500, -1)],
    (6, frozenset({300, 800, 801, 802, 803, 700})):
        [(300, 800, 5), (800, 801, 4), (801, 802, 3), (802, 803, 2), (803, 700, 1)],

    # 6-LANE SCENARIOS
    (7, frozenset({100, 101, 102, 200, 201, 202, 300})):
        [(100, 101, -3), (101, 102, -2), (102, 200, -1), (201, 200, 1), (202, 201, 2), (300, 202, 3)],
    (7, frozenset({100, 400, 401, 402, 403, 404, 500})):
        [(100, 400, -6), (400, 401, -5), (401, 402, -4), (402, 403, -3), (403, 404, -2), (404, 500, -1)],
    (7, frozenset({300, 800, 801, 802, 803, 804, 700})):
        [(300, 800, 6), (800, 801, 5), (801, 802, 4), (802, 803, 3), (803, 804, 2), (804, 700, 1)],
    (7, frozenset({100, 101, 102, 103, 200, 201, 300})):
        [(100, 101, -4), (101, 102, -3), (102, 103, -2), (103, 200, -1), (201, 200, 1), (300, 201, 2)],
    (7, frozenset({100, 101, 200, 201, 202, 203, 300})):
        [(100, 101, -2), (101, 200, -1), (201, 200, 1), (202, 201, 2), (203, 202, 3), (300, 203, 4)],

    # 7-LANE SCENARIOS
    (8, frozenset({100, 101, 102, 200, 201, 202, 203, 300})):
        [(100, 101, -3), (101, 102, -2), (102, 200, -1), (201, 200, 1), (202, 201, 2),
         (203, 202, 3), (300, 203, 4)],
    (8, frozenset({100, 101, 102, 103, 200, 201, 202, 300})):
        [(100, 101, -4), (101, 102, -3), (102, 103, -2), (103, 200, -1), (201, 200, 1),
         (202, 201, 2), (300, 202, 3)],
}

# Stop line like road markings that are converted as signals
STOP_LINE_TYPES = {"de294": "294", "de341": "341"}


def normalize_angle(angle):
    """Normalize an angle to [-pi, pi)."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def to_int(value):
    """Convert ids like 12, 12.0 or '12.0' to int; None if not convertible."""
    try:
        if value is None or str(value).strip() in ("", "None", "nan"):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def normalize_re_id(value):
    """re_id as string: '27', 27 and 27.0 -> '27', 're_27' -> 're_27'; None if empty."""
    if value is None or str(value).strip() in ("", "None", "nan"):
        return None
    as_int = to_int(value)
    return str(as_int) if as_int is not None else str(value).strip()


def split_ids(value):
    """'re_1, re_2' -> ['re_1', 're_2']"""
    ids = [normalize_re_id(v) for v in str(value or "").split(",")]
    return [i for i in ids if i is not None]


def centerline_id_for(right_way, left_way):
    """Centerline way_id of a lane, same rule as in the Lanelet2 converter."""
    return int(f"{min(int(str(right_way)[:3]), int(str(left_way)[:3]))}12")


class GeoJSONToOpenDriveConverter:
    # Boundary ways that define the lane layout (same set as the Lanelet2 converter)
    BOUNDARY_LANE_TYPES = ["road", "road_cycle", "cycle", "tram", "rail"]
    BICYCLE_LANE_TYPES = {"road_cycle", "cycle"}
    # Boundary ways that may be used as OpenDRIVE reference line
    REFERENCE_LANE_TYPES = ["road", "road_cycle", "cycle"]
    REFERENCE_PRIORITY = [200, 202, 500, 700]
    # ISO 3166-1 alpha-2 country code and VzKat revision for traffic signs
    SIGN_COUNTRY = "DE"
    SIGN_COUNTRY_REVISION = "2017"
    # Traffic lights have no official StVO type; they use the generic
    # ASAM OpenDRIVE Signal Reference catalog (country="OpenDRIVE")
    LIGHT_COUNTRY = "OpenDRIVE"

    def __init__(self, crs_from="EPSG:4326", crs_to="EPSG:25833", origin=None,
                 physical_position_tolerance=0.05, inertial_for_all_signals=False):
        """
        crs_from:
            CRS of the GeoJSON coordinates. EPSG:4326 = WGS84 geographic (lon/lat).
        crs_to:
            Projected metric CRS. EPSG:25833 = ETRS89 / UTM zone 33N.
        origin:
            Optional (x, y) in crs_to used as local origin. All OpenDRIVE
            coordinates are written relative to it and it is stored in the
            header <offset>. If None, the rounded bounding box center is used.
        physical_position_tolerance:
            If the logical position (s, t) of a signal deviates from its
            physical GeoJSON position by more than this [m], e.g. because the
            signal lies outside the reference line, or because a traffic light
            is placed at its stop line, the exact physical position is written
            as <positionInertial> (ASAM OpenDRIVE 1.7+).
        inertial_for_all_signals:
            If True, <positionInertial> is written for every signal.

        Lane assignment:
            - centerline traffic_rule  -> the lane of that centerline
            - cycle boundary traffic_rule -> the bicycle lane bounded by that way
            - pedestrian_marking traffic_rule -> the crosswalk (pedestrian lights)
            - stop lines (ref_lines) inherit the lanes of the elements referencing them
            OpenDRIVE lane ids are |lane_no| from the layout table, negative if the
            lane drives in +s direction of the reference line, positive otherwise.
        """
        self.crs_to = crs_to
        self.transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)
        self.origin = origin
        self.physical_position_tolerance = physical_position_tolerance
        self.inertial_for_all_signals = inertial_for_all_signals

        self.road_ways = {}          # road_id -> {way_id: LineString} (all boundary types)
        self.way_types = {}          # (road_id, way_id) -> lane_type
        self.reference_lines = {}    # road_id -> LineString
        self.traffic_rules = {}      # re_id -> [road_id, ...]
        self.re_lanes = {}           # re_id -> [(road_id, centerline_way_id), ...]
        self.re_crosswalks = {}      # re_id -> {crosswalk road_id, ...}
        self.crosswalk_lines = {}    # crosswalk road_id -> [LineString, ...]
        self.regulatory_lines = {}   # re_id -> LineString
        self.regulatory_props = {}   # re_id -> GeoJSON properties
        self.stop_line_ids = set()   # re_ids referenced in any ref_lines column
        self.spatial_index = None
        self.road_id_list = []

        self.light_mapping = {
            "pedestrian": {"type": "1000002", "subtype": "-1"},
            "cycle": {"type": "1000013", "subtype": "-1"},  # on geojson maps every subtype used as 'cycle' tag

            "red_yellow_green": {"type": "1000001", "subtype": "-1"},
            "red_yellow":       {"type": "1000009", "subtype": "10"},

            "arrow_left_green": {"type": "1000012", "subtype": "10"},
            "arrow_left_red_yellow_green":  {"type": "1000011", "subtype": "10"},
            "arrow_right_red_yellow_green": {"type": "1000011", "subtype": "20"},
            # "arrow_right_red_green":        {"type": "", "subtype": ""},
            "arrow_right_yellow_green":     {"type": "1000010", "subtype": "20"}
        }

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------
    def _compute_origin(self, geojson_data):
        """Rounded bounding box center of all features in crs_to."""
        min_x = min_y = math.inf
        max_x = max_y = -math.inf

        def visit(coords):
            nonlocal min_x, min_y, max_x, max_y
            if not coords:
                return
            if isinstance(coords[0], (int, float)):
                x, y = self.transformer.transform(coords[0], coords[1])
                min_x, max_x = min(min_x, x), max(max_x, x)
                min_y, max_y = min(min_y, y), max(max_y, y)
            else:
                for c in coords:
                    visit(c)

        for feature in geojson_data.get("features", []):
            visit(feature.get("geometry", {}).get("coordinates", []))

        if math.isinf(min_x):
            return 0.0, 0.0
        return float(round((min_x + max_x) / 2.0)), float(round((min_y + max_y) / 2.0))

    def _project(self, lon, lat):
        """Project lon/lat to local coordinates (crs_to minus origin)."""
        x, y = self.transformer.transform(lon, lat)
        return x - self.origin[0], y - self.origin[1]

    def _project_line(self, coords):
        return LineString([self._project(c[0], c[1]) for c in coords])

    def _geo_reference_string(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proj_str = CRS.from_user_input(self.crs_to).to_proj4()
        return proj_str.replace("+type=crs", "").strip()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _tangent_at(line, s):
        """Heading of a LineString at arc length s (in its drawing direction)."""
        if s + 0.1 >= line.length:
            a = line.interpolate(max(0.0, s - 0.1))
            b = line.interpolate(s)
        else:
            a = line.interpolate(s)
            b = line.interpolate(s + 0.1)
        return math.atan2(b.y - a.y, b.x - a.x)

    def _calculate_s_t_hdg(self, point, ref_line):
        """
        s, t and reference line heading for a point. Positive t is left.

        If the point lies before the start or after the end of the reference
        line, s is clamped to 0 or length. In that case t is the lateral
        offset along the normal at that end (not the oblique distance to the
        end point), so (s, t) is the closest valid logical position.

        Returns (s, t, road_tangent, proj_point, logical_point) where
        logical_point is the position (s, t) converted back to x/y.
        """
        s = ref_line.project(point)
        proj_point = ref_line.interpolate(s)
        road_tangent = self._tangent_at(ref_line, s)

        dx_point = point.x - proj_point.x
        dy_point = point.y - proj_point.y
        normal_x, normal_y = -math.sin(road_tangent), math.cos(road_tangent)

        if 1e-6 < s < ref_line.length - 1e-6:
            dist = ref_line.distance(point)
            cross_product = math.cos(road_tangent) * dy_point - math.sin(road_tangent) * dx_point
            t = dist if cross_product > 0 else -dist
        else:
            # Clamped: keep only the lateral component
            t = dx_point * normal_x + dy_point * normal_y

        logical_point = Point(proj_point.x + t * normal_x, proj_point.y + t * normal_y)
        return s, t, road_tangent, proj_point, logical_point

    # ------------------------------------------------------------------
    # Lanes from way_ids
    # ------------------------------------------------------------------
    def _layout(self, road_id):
        ways = self.road_ways.get(road_id)
        if not ways:
            return None
        return LANE_LAYOUTS.get((len(ways), frozenset(ways.keys())))

    def _lane_info(self, road_id, centerline_way_id):
        """(right_boundary, left_boundary, lane_no) of a lane, or None."""
        layout = self._layout(road_id)
        if not layout or centerline_way_id is None:
            return None
        ways = self.road_ways[road_id]
        for right_way, left_way, lane_no in layout:
            if centerline_id_for(right_way, left_way) == centerline_way_id:
                return ways[right_way], ways[left_way], lane_no
        return None

    def _bicycle_lane_for_way(self, road_id, way_id):
        """Centerline way_id of the bicycle lane bounded by a cycle way, or None."""
        layout = self._layout(road_id)
        if not layout:
            return None
        candidates = [(r, l) for r, l, _ in layout if way_id in (r, l)]
        bicycle = [(r, l) for r, l in candidates
                   if {self.way_types.get((road_id, r)), self.way_types.get((road_id, l))} <= self.BICYCLE_LANE_TYPES]
        chosen = bicycle or candidates
        if len(chosen) != 1:
            return None
        return centerline_id_for(*chosen[0])

    def _travel_heading_from_bounds(self, right_line, left_line, point):
        """
        Local travel heading of a lane near a point: the direction in which
        right_line is on the right and left_line is on the left.
        Returns (heading, lane_point) where lane_point is the lane middle.
        """
        s = right_line.project(point)
        p_right = right_line.interpolate(s)
        heading = self._tangent_at(right_line, s)

        p_left = left_line.interpolate(left_line.project(p_right))
        cross = math.cos(heading) * (p_left.y - p_right.y) - math.sin(heading) * (p_left.x - p_right.x)
        if cross < 0:
            heading += math.pi

        lane_point = Point((p_right.x + p_left.x) / 2.0, (p_right.y + p_left.y) / 2.0)
        return normalize_angle(heading), lane_point

    def _signal_lanes(self, re_id, road_id, point):
        """
        Lanes on road_id that the element is assigned to, evaluated where the
        point projects onto the reference line of that road.
        Returns a list of (opendrive_lane_id, travel_heading, lane_point).
        """
        ref = self.reference_lines[road_id]
        s = ref.project(point)
        foot = ref.interpolate(s)
        tangent = self._tangent_at(ref, s)

        lanes = {}
        for r_id, cl_id in self.re_lanes.get(re_id, []):
            if r_id != road_id:
                continue
            info = self._lane_info(road_id, cl_id)
            if info is None:
                continue
            right_line, left_line, lane_no = info
            travel, lane_point = self._travel_heading_from_bounds(right_line, left_line, foot)
            lane_id = -abs(lane_no) if math.cos(travel - tangent) >= 0 else abs(lane_no)
            lanes[lane_id] = (lane_id, travel, lane_point)
        return list(lanes.values())

    @staticmethod
    def _orientation_from_lanes(lane_ids):
        """'+' if all lanes drive in +s (negative ids), '-' if all in -s, 'none' if both."""
        if all(i < 0 for i in lane_ids):
            return "+"
        if all(i > 0 for i in lane_ids):
            return "-"
        return "none"

    @staticmethod
    def _add_validity(node, lane_ids):
        """One <validity> per contiguous lane range on each side of the reference line."""
        for side in ([i for i in lane_ids if i < 0], [i for i in lane_ids if i > 0]):
            ordered = sorted(side, key=abs)
            ranges = []
            for lane_id in ordered:
                if ranges and abs(lane_id) == abs(ranges[-1][-1]) + 1:
                    ranges[-1].append(lane_id)
                else:
                    ranges.append([lane_id])
            for r in ranges:
                ET.SubElement(node, "validity", fromLane=str(min(r)), toLane=str(max(r)))

    @staticmethod
    def _fallback_travel(ref_tangent, t):
        """
        Guess the served travel direction from the road side (right-hand traffic):
        signals right of the reference line (t <= 0) serve +s traffic.
        """
        return normalize_angle(ref_tangent if t <= 0 else ref_tangent + math.pi)

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_sign_type_subtype(sign_str):
        """
        Split a German VzKat sign number into OpenDRIVE type and subtype.

        A dot is part of the sign number and stays in the type, a dash
        separates the variant (subtype):
            de274.1     -> type "274.1", subtype "-1"  (Tempo 30 zone begin)
            de274.2     -> type "274.2", subtype "-1"  (Tempo 30 zone end)
            de274.1-20  -> type "274.1", subtype "20"  (Tempo 20 zone begin)
        """
        val = str(sign_str).strip().lower()
        if val.startswith("de:"):
            val = val[3:]
        elif val.startswith("de"):
            val = val[2:]

        if "-" in val:
            sign_type, subtype = val.split("-", 1)
            return sign_type, subtype if subtype else "-1"
        return val, "-1"

    def _calculate_sign_geometry(self, coords):
        """
        The GeoJSON line is the sign plate (or the stop line) seen from above.
        Returns center (Point), axis heading [rad] and length [m].
        The drawing direction of the line is irrelevant.
        """
        x1, y1 = self._project(coords[0][0], coords[0][1])
        x2, y2 = self._project(coords[-1][0], coords[-1][1])
        plate_heading = math.atan2(y2 - y1, x2 - x1)
        plate_length = math.hypot(x2 - x1, y2 - y1)
        center = Point((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        return center, plate_heading, plate_length

    @staticmethod
    def _resolve_facing(center, plate_heading, plate_length, travel_heading, lane_point, parallel_limit=0.25):
        """
        Pick the plate normal that looks at the served traffic.
        - Normally the normal that points against the travel direction (towards oncoming traffic).
        - If the plate is (almost) parallel to the travel direction (|cos| < parallel_limit,
          i.e. within ~15 deg, e.g. parking signs), or no travel direction is known
          (pedestrian lights), the normal that points towards the lane / target point.
        """
        normals = (plate_heading + math.pi / 2, plate_heading - math.pi / 2)
        if travel_heading is not None:
            oncoming = travel_heading + math.pi
            if plate_length < 0.05:
                # Degenerate plate, assume it faces oncoming traffic directly
                return normalize_angle(oncoming)
            along = math.cos(normals[0] - oncoming)
            if abs(along) >= parallel_limit:
                return normalize_angle(normals[0] if along > 0 else normals[1])

        dx, dy = lane_point.x - center.x, lane_point.y - center.y
        towards = [math.cos(n) * dx + math.sin(n) * dy for n in normals]
        return normalize_angle(normals[0] if towards[0] >= towards[1] else normals[1])

    @staticmethod
    def _h_offset(facing, orientation, ref_tangent):
        """
        OpenDRIVE hOffset:
            orientation="+":    face points to -s at hOffset=0, offset is CCW from -s.
            orientation="-":    face points to +s at hOffset=0, offset is CCW from +s.
            orientation="none": offset is relative to the reference line (+s).
        """
        base = ref_tangent + math.pi if orientation == "+" else ref_tangent
        return normalize_angle(facing - base)

    def _stop_line_point(self, props, re_id):
        """
        Logical position of a traffic light: the middle of the stop line whose
        re_id is given in the ref_lines column. Returns (stop_line_id, Point) or (None, None).
        """
        ref_ids = split_ids(props.get("ref_lines"))
        if not ref_ids:
            return None, None
        if len(ref_ids) > 1:
            print(f"[WARN] re_id={re_id}: several ref_lines {ref_ids}, using {ref_ids[0]}")
        stop_line = self.regulatory_lines.get(ref_ids[0])
        if stop_line is None:
            print(f"[WARN] re_id={re_id}: stop line {ref_ids[0]} not found, using light position as s")
            return None, None
        return ref_ids[0], stop_line.interpolate(0.5, normalized=True)

    def _crosswalk_center(self, re_id):
        """Center of the crosswalk(s) a pedestrian light is assigned to, or None."""
        mids = [line.interpolate(0.5, normalized=True)
                for cw in self.re_crosswalks.get(re_id, ()) for line in self.crosswalk_lines.get(cw, [])]
        if not mids:
            return None
        return Point(sum(p.x for p in mids) / len(mids), sum(p.y for p in mids) / len(mids))

    # ------------------------------------------------------------------
    # Collecting input
    # ------------------------------------------------------------------
    def _build_reference_lines_and_index(self, geojson_data):
        reference_candidates = {}

        for feature in geojson_data.get("features", []):
            props = feature.get("properties", {})
            lane_type = props.get("lane_type")
            coords = feature.get("geometry", {}).get("coordinates", [])

            if lane_type == "pedestrian_marking":
                road_id = to_int(props.get("road_id"))
                if road_id is not None and len(coords) >= 2:
                    self.crosswalk_lines.setdefault(road_id, []).append(self._project_line(coords))
                continue

            if lane_type not in self.BOUNDARY_LANE_TYPES:
                continue

            road_id = to_int(props.get("road_id"))
            way_id = to_int(props.get("way_id"))
            if road_id is None or way_id is None or len(coords) < 2:
                continue

            line = self._project_line(coords)
            self.road_ways.setdefault(road_id, {})[way_id] = line
            self.way_types[(road_id, way_id)] = lane_type
            if lane_type in self.REFERENCE_LANE_TYPES:
                reference_candidates.setdefault(road_id, {})[way_id] = line

        line_geometries = []
        for road_id, ways in reference_candidates.items():
            ref_way = next((w for w in self.REFERENCE_PRIORITY if w in ways), min(ways.keys()))
            self.reference_lines[road_id] = ways[ref_way]
            line_geometries.append(ways[ref_way])
            self.road_id_list.append(road_id)

        if line_geometries:
            self.spatial_index = STRtree(line_geometries)

    def _assign(self, re_id, road_id, centerline_way_id):
        if road_id not in self.traffic_rules.setdefault(re_id, []):
            self.traffic_rules[re_id].append(road_id)
        lane_key = (road_id, centerline_way_id)
        if centerline_way_id is not None and lane_key not in self.re_lanes.setdefault(re_id, []):
            self.re_lanes[re_id].append(lane_key)

    def _extract_traffic_rules(self, geojson_data):
        """Read traffic_rule from centerlines, cycle lane boundaries and crosswalks."""
        for feature in geojson_data.get("features", []):
            props = feature.get("properties", {})
            lane_type = props.get("lane_type")
            re_ids = split_ids(props.get("traffic_rule"))
            road_id = to_int(props.get("road_id"))
            if not re_ids or road_id is None:
                continue

            if lane_type == "centerline":
                centerline_way_id = to_int(props.get("way_id"))
                for r_id in re_ids:
                    self._assign(r_id, road_id, centerline_way_id)

            elif lane_type in self.BICYCLE_LANE_TYPES:
                way_id = to_int(props.get("way_id"))
                centerline_way_id = self._bicycle_lane_for_way(road_id, way_id)
                if centerline_way_id is None:
                    print(f"[WARN] {re_ids}: bicycle lane for road {road_id} way {way_id} not found")
                for r_id in re_ids:
                    self._assign(r_id, road_id, centerline_way_id)

            elif lane_type == "pedestrian_marking":
                for r_id in re_ids:
                    self.re_crosswalks.setdefault(r_id, set()).add(road_id)

    def _collect_regulatory_lines(self, geojson_data):
        """Store regulatory_element lines and properties by re_id, and collect stop line ids."""
        for feature in geojson_data.get("features", []):
            props = feature.get("properties", {})
            if props.get("lane_type") != "regulatory_element":
                continue
            re_id = normalize_re_id(props.get("re_id"))
            coords = feature.get("geometry", {}).get("coordinates", [])
            if re_id is None or len(coords) < 2:
                continue
            self.regulatory_lines[re_id] = self._project_line(coords)
            self.regulatory_props[re_id] = props
            self.stop_line_ids.update(split_ids(props.get("ref_lines"))[:1])

    def _propagate_stop_line_lanes(self):
        """Stop lines inherit the roads and lanes of the elements that reference them."""
        for re_id, props in self.regulatory_props.items():
            ref_ids = split_ids(props.get("ref_lines"))
            if not ref_ids:
                continue
            stop_id = ref_ids[0]
            for road_id in self.traffic_rules.get(re_id, []):
                if road_id not in self.traffic_rules.setdefault(stop_id, []):
                    self.traffic_rules[stop_id].append(road_id)
            for lane_key in self.re_lanes.get(re_id, []):
                if lane_key not in self.re_lanes.setdefault(stop_id, []):
                    self.re_lanes[stop_id].append(lane_key)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    def _create_base_skeleton(self, root):
        for road_id, ref_line in self.reference_lines.items():
            road_node = ET.SubElement(root, "road", id=str(road_id), length=f"{ref_line.length:.4f}",
                                      junction="-1", name=f"Road_{road_id}")
            plan_view = ET.SubElement(road_node, "planView")

            coords_list = list(ref_line.coords)
            current_s = 0.0

            for i in range(len(coords_list) - 1):
                p1, p2 = coords_list[i], coords_list[i + 1]
                dx, dy = p2[0] - p1[0], p2[1] - p1[1]
                seg_len = math.hypot(dx, dy)
                hdg = math.atan2(dy, dx)
                geom = ET.SubElement(plan_view, "geometry", s=f"{current_s:.4f}", x=f"{p1[0]:.4f}",
                                     y=f"{p1[1]:.4f}", hdg=f"{hdg:.4f}", length=f"{seg_len:.4f}")
                ET.SubElement(geom, "line")
                current_s += seg_len

            lanes = ET.SubElement(road_node, "lanes")
            lane_sec = ET.SubElement(lanes, "laneSection", s="0.0")
            ET.SubElement(ET.SubElement(lane_sec, "center"), "lane", id="0", type="none", level="false")
            ET.SubElement(road_node, "signals")

    def _choose_primary_road(self, re_id, point):
        associated = [r for r in self.traffic_rules.get(re_id, []) if r in self.reference_lines]
        if associated:
            return min(associated, key=lambda r: self.reference_lines[r].distance(point)), associated
        if self.spatial_index is None:
            return None, []
        nearest = self.road_id_list[self.spatial_index.nearest(point)]
        return nearest, [nearest]

    def load_and_convert(self, geojson_data):
        if self.origin is None:
            self.origin = self._compute_origin(geojson_data)

        self._build_reference_lines_and_index(geojson_data)
        self._extract_traffic_rules(geojson_data)
        self._collect_regulatory_lines(geojson_data)
        self._propagate_stop_line_lanes()

        root = ET.Element("OpenDRIVE")
        # 1.7 is required for <positionInertial> in signals
        header = ET.SubElement(root, "header", revMajor="1", revMinor="7", name="MotorAI_Converted_Map",
                               version="1.0", date=datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
        geo_ref = ET.SubElement(header, "geoReference")
        geo_ref.text = f"<![CDATA[{self._geo_reference_string()}]]>"
        # projected (crs_to) = local + offset
        ET.SubElement(header, "offset", x=f"{self.origin[0]:.4f}", y=f"{self.origin[1]:.4f}",
                      z="0.0", hdg="0.0")

        self._create_base_skeleton(root)

        written = set()
        light_groups = {}  # stop line re_id -> [(light re_id, line_sub)]
        crosswalk_groups = {}  # crosswalk (pedestrian_marking) road_id -> [(light re_id, line_sub)]

        for feature in geojson_data.get("features", []):
            props = feature.get("properties", {})
            if props.get("lane_type") != "regulatory_element":
                continue

            line_type = props.get("line_type")
            line_sub = str(props.get("line_sub", "")).lower()
            re_id = normalize_re_id(props.get("re_id"))
            coords = feature.get("geometry", {}).get("coordinates", [])
            if re_id is None or len(coords) < 2:
                continue

            # Stop lines: de294 always, de341 only if it is the stop line of a traffic light/sign
            is_stop_line = line_sub in STOP_LINE_TYPES
            if line_sub == "de341" and re_id not in self.stop_line_ids:
                continue
            is_light = line_type == "traffic_light" and not is_stop_line
            is_pedestrian_light = is_light and line_sub == "pedestrian"

            # Physical position of the sign plate / light head / stop line
            signal_point, plate_heading, plate_length = self._calculate_sign_geometry(coords)

            # Logical position: where the validity starts.
            # Traffic lights: the stop line (ref_lines). Everything else: the element itself.
            logical_source = signal_point
            stop_id = None
            if is_light:
                stop_id, stop_point = self._stop_line_point(props, re_id)
                if stop_point is not None:
                    logical_source = stop_point
                    light_groups.setdefault(stop_id, []).append((re_id, line_sub))

            crosswalk_center = self._crosswalk_center(re_id) if is_pedestrian_light else None
            if is_pedestrian_light and stop_id is None and self.re_crosswalks.get(re_id):
                # Pedestrian lights of the same crosswalk switch together
                crosswalk_ids = sorted(self.re_crosswalks[re_id])
                if len(crosswalk_ids) > 1:
                    print(f"[WARN] re_id={re_id}: assigned to several crosswalks {crosswalk_ids}, "
                          f"controller of crosswalk {crosswalk_ids[0]} is used")
                crosswalk_groups.setdefault(crosswalk_ids[0], []).append((re_id, line_sub))
            primary_road_id, associated_roads = self._choose_primary_road(
                re_id, crosswalk_center if crosswalk_center is not None else logical_source)
            if primary_road_id is None:
                continue
            ref_roads = [r for r in associated_roads if r != primary_road_id]
            signals_node = root.find(f"./road[@id='{primary_road_id}']").find("signals")

            s, t, ref_tangent, ref_proj_point, logical_point = self._calculate_s_t_hdg(
                logical_source, self.reference_lines[primary_road_id])
            clamp_deviation = logical_source.distance(logical_point)
            physical_deviation = signal_point.distance(logical_point)

            # Lanes, orientation and facing
            lanes = self._signal_lanes(re_id, primary_road_id, logical_source)
            lane_ids = sorted(l[0] for l in lanes)
            if crosswalk_center is not None:
                # Pedestrian light: valid for the crosswalk, faces the crossing
                orientation_val = "none"
                facing = self._resolve_facing(signal_point, plate_heading, plate_length, None, crosswalk_center)
            elif lanes:
                orientation_val = self._orientation_from_lanes(lane_ids)
                _, travel, lane_point = min(lanes, key=lambda l: l[2].distance(signal_point))
                facing = self._resolve_facing(signal_point, plate_heading, plate_length, travel, lane_point)
            else:
                if self.re_lanes.get(re_id) or self.re_crosswalks.get(re_id):
                    print(f"[WARN] re_id={re_id}: assigned lanes could not be resolved on road "
                          f"{primary_road_id}, using road side fallback")
                travel = self._fallback_travel(ref_tangent, t)
                orientation_val = "+" if travel == normalize_angle(ref_tangent) else "-"
                facing = self._resolve_facing(signal_point, plate_heading, plate_length, travel, ref_proj_point)
            h_offset = self._h_offset(facing, orientation_val, ref_tangent)

            elevation = props.get("elevation", "5.0") or "5.0"
            if is_stop_line:
                elevation, height = "0.0", "0.0"
                width = f"{plate_length:.2f}"
            else:
                height = "0.5"
                # Use the annotated plate length as width if it is plausible
                width = f"{plate_length:.2f}" if 0.2 <= plate_length <= 3.0 else "0.5"

            # Create XML Node
            signal_node = ET.Element("signal")
            signal_node.set("s", f"{s:.4f}")
            signal_node.set("t", f"{t:.4f}")
            signal_node.set("id", re_id)
            signal_node.set("name", str(line_sub))
            signal_node.set("dynamic", "yes" if is_light else "no")
            signal_node.set("orientation", orientation_val)
            signal_node.set("zOffset", str(elevation))

            if is_stop_line:
                signal_node.set("country", self.SIGN_COUNTRY)
                signal_node.set("countryRevision", self.SIGN_COUNTRY_REVISION)
                signal_node.set("type", STOP_LINE_TYPES[line_sub])
                signal_node.set("subtype", "-1")
            elif line_type == "traffic_sign":
                t_type, t_subtype = self._parse_sign_type_subtype(line_sub)
                signal_node.set("country", self.SIGN_COUNTRY)
                signal_node.set("countryRevision", self.SIGN_COUNTRY_REVISION)
                signal_node.set("type", t_type)
                signal_node.set("subtype", t_subtype)
            else:
                mapped = self.light_mapping.get(line_sub, {"type": "1000001", "subtype": "-1"})
                signal_node.set("country", self.LIGHT_COUNTRY)
                signal_node.set("type", mapped["type"])
                signal_node.set("subtype", mapped["subtype"])

            signal_node.set("height", str(height))
            signal_node.set("width", str(width))
            # OpenDRIVE uses hOffset for signals (hdg is only valid for objects)
            signal_node.set("hOffset", f"{h_offset:.4f}")
            signal_node.set("pitch", "0.0")
            signal_node.set("roll", "0.0")

            # Child order (ASAM): validity, dependency, reference, positionInertial
            self._add_validity(signal_node, lane_ids)
            if stop_id is not None:
                ET.SubElement(signal_node, "reference", elementType="signal", elementId=stop_id, type="stopLine")

            # Physical position, if it deviates from the logical position (s, t).
            if self.inertial_for_all_signals or physical_deviation > self.physical_position_tolerance:
                ET.SubElement(signal_node, "positionInertial",
                              x=f"{signal_point.x:.4f}", y=f"{signal_point.y:.4f}",
                              z=f"{float(elevation):.4f}", hdg=f"{facing:.4f}",
                              pitch="0.0", roll="0.0")
            if clamp_deviation > self.physical_position_tolerance:
                if s <= 1e-6 or s >= self.reference_lines[primary_road_id].length - 1e-6:
                    reason = f"outside the reference line (s clamped to {s:.2f})"
                else:
                    reason = f"at a reference line vertex/kink (s={s:.2f})"
                print(f"[INFO] re_id={re_id}: logical position on road {primary_road_id} deviates by "
                      f"{clamp_deviation:.2f} m, {reason}")

            signals_node.append(signal_node)
            written.add(re_id)

            # Signal references on the other assigned roads, with their own lanes
            for other_road_id in ref_roads:
                other_signals_node = root.find(f"./road[@id='{other_road_id}']").find("signals")
                ref_s, ref_t, other_tangent, _, _ = self._calculate_s_t_hdg(
                    logical_source, self.reference_lines[other_road_id])

                other_lanes = self._signal_lanes(re_id, other_road_id, logical_source)
                other_ids = sorted(l[0] for l in other_lanes)
                if other_ids:
                    other_orientation = self._orientation_from_lanes(other_ids)
                else:
                    travel = self._fallback_travel(other_tangent, ref_t)
                    other_orientation = "+" if travel == normalize_angle(other_tangent) else "-"

                sig_ref_node = ET.Element("signalReference")
                sig_ref_node.set("s", f"{ref_s:.4f}")
                sig_ref_node.set("t", f"{ref_t:.4f}")
                sig_ref_node.set("id", re_id)
                sig_ref_node.set("orientation", other_orientation)
                self._add_validity(sig_ref_node, other_ids)
                other_signals_node.append(sig_ref_node)

        # Controllers:
        #   - lights sharing the same stop line (ref_lines) switch together
        #   - pedestrian lights assigned to the same crosswalk (pedestrian_marking road_id) switch together
        controllers = [(f"ctrl_{stop_id}", f"stop_line_{stop_id}", members)
                       for stop_id, members in light_groups.items()]
        controllers += [(f"ctrl_crosswalk_{cw_id}", f"crosswalk_{cw_id}", members)
                        for cw_id, members in crosswalk_groups.items()]
        grouped = set()
        for ctrl_id, name, members in controllers:
            controller = ET.SubElement(root, "controller", id=ctrl_id, name=name)
            for light_id, light_sub in members:
                ET.SubElement(controller, "control", signalId=light_id, type=light_sub)
                grouped.add(light_id)
        ungrouped = sorted(
            r for r, p in self.regulatory_props.items()
            if r in written and r not in grouped and p.get("line_type") == "traffic_light"
            and str(p.get("line_sub", "")).lower() not in STOP_LINE_TYPES)
        print(f"[INFO] {len(light_groups)} stop line controllers and {len(crosswalk_groups)} crosswalk "
              f"controllers written; traffic lights without controller: {ungrouped if ungrouped else 'none'}")

        xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
        xmlstr = xmlstr.replace("&lt;![CDATA[", "<![CDATA[").replace("]]&gt;", "]]>")
        return xmlstr


# =========================================================
if __name__ == "__main__":

    input_geojson = '/home/sibas@motor-ai.local/Documents/Simulation/Berlin002_traffic_elements.geojson'
    output_xodr = '/home/sibas@motor-ai.local/Documents/Simulation/Berlin002_traffic_elements.xodr'

    # Read the GeoJSON file
    with open(input_geojson, 'r', encoding='utf-8') as f:
        geojson_dict = json.load(f)

    # GeoJSON lon/lat is WGS84 (EPSG:4326), projected to ETRS89 / UTM 33N (EPSG:25833)
    converter = GeoJSONToOpenDriveConverter(crs_from="EPSG:4326", crs_to="EPSG:25833")

    # Call the load_and_convert method via the class instance
    xodr_string = converter.load_and_convert(geojson_dict)

    # Export the xodr file
    with open(output_xodr, 'w', encoding='utf-8') as f:
        f.write(xodr_string)

    print(f" OpenDRIVE file exported to: {output_xodr}")
