import sys
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
import h3
import pycountry
import reverse_geocoder as rg

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
# DATA_DIR = Path("/Users/mjsteenberg/11 Nov 25 - App Opens Data VoiceMap")
PORTS_PATH = DATA_DIR / "world_ports.csv"
DEFAULT_DATA_PATHS = (
    DATA_DIR / "voicemap_app_opens_master.parquet",
    DATA_DIR / "voicemap_app_opens_master.csv",
)

DEFAULT_H3_RESOLUTION = 6
DEFAULT_COLOR_PERCENTILE = 96
DEFAULT_MAX_RADIUS = 5000
GLOBAL_VIEW_ZOOM_THRESHOLD = 2.5
GLOBAL_COLOR_PERCENTILE = 99
GLOBAL_RADIUS_SCALE = 0.4
GLOBAL_MIN_RADIUS = 1500
MIN_H3_RESOLUTION = 4
MAX_H3_RESOLUTION = 9
MAX_POINT_ROWS = 300_000
POINT_LAYER_HEIGHT = 650
CLUSTER_LAYER_HEIGHT = 650
MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

LOCATION_CACHE: dict[str, dict[str, str]] = {}


def discover_data_path() -> Path:
    """Return the first existing master dataset path or exit with an error."""
    candidate = st.sidebar.text_input("Dataset path", value=str(DEFAULT_DATA_PATHS[0]))
    path = Path(candidate).expanduser().resolve()

    if not path.exists():
        fallback = next((p for p in DEFAULT_DATA_PATHS if p.exists()), None)
        if fallback is None:
            st.error(
                "No dataset found. Please export `voicemap_app_opens_master` via the notebook "
                "and provide its path above."
            )
            st.stop()
        st.sidebar.warning(f"Using fallback dataset: {fallback}")
        return fallback

    return path


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    """Load the master dataset from Parquet or CSV and normalise the date column."""
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        st.error("Unsupported file type. Use Parquet or CSV.")
        st.stop()

    df["Created at"] = pd.to_datetime(df["Created at"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["Created at", "Lat", "Lng"])
    return df


@st.cache_data(show_spinner=False)
def load_ports(path: Path = PORTS_PATH) -> pd.DataFrame:
    """Load world ports metadata with latitude/longitude and labels."""
    if not path.exists():
        return pd.DataFrame(columns=["latitude", "longitude", "port_label"])

    ports = pd.read_csv(path)
    lat_col = "Latitude"
    lng_col = "Longitude"
    ports = ports[["Region Name", "Main Port Name", lat_col, lng_col]].rename(
        columns={lat_col: "latitude", lng_col: "longitude"}
    )
    ports.dropna(subset=["latitude", "longitude"], inplace=True)
    ports["latitude"] = pd.to_numeric(ports["latitude"], errors="coerce")
    ports["longitude"] = pd.to_numeric(ports["longitude"], errors="coerce")
    ports.dropna(subset=["latitude", "longitude"], inplace=True)
    ports["port_label"] = ports.apply(
        lambda row: ", ".join(
            [
                part
                for part in [row.get("Main Port Name", ""), row.get("Region Name", "")] 
                if part and isinstance(part, str)
            ]
        )
        or f"Port @ {row['latitude']:.2f}, {row['longitude']:.2f}",
        axis=1,
    )
    return ports


def filter_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """Render a date range selector and return the filtered dataframe."""
    min_date = df["Created at"].min()
    max_date = df["Created at"].max()

    if min_date is None or max_date is None:
        st.warning("Dataset has no valid dates after parsing.")
        return df

    preferred_start = date(2025, 3, 1)
    preferred_end = date(2025, 11, 11)
    dataset_start = min_date.date()
    dataset_end = max_date.date()
    start_default = max(dataset_start, preferred_start)
    end_default = min(dataset_end, preferred_end)
    if start_default > end_default:
        start_default, end_default = dataset_start, dataset_end

    start_date, end_date = st.sidebar.date_input(
        "Date range",
        value=(start_default, end_default),
        min_value=dataset_start,
        max_value=dataset_end,
    )

    if isinstance(start_date, tuple) or isinstance(end_date, tuple):
        st.error("Please select a valid date range.")
        st.stop()

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if end_ts < start_ts:
        st.sidebar.error("End date must be on or after start date.")
        st.stop()

    mask = (df["Created at"] >= start_ts) & (df["Created at"] <= end_ts)
    filtered = df.loc[mask].copy()
    return filtered


def sample_points(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, int, bool]:
    """Return a subset of points limited to MAX_POINT_ROWS for rendering."""
    subset = df[columns].copy()
    total = len(subset)
    sampled = False
    if total > MAX_POINT_ROWS:
        subset = subset.sample(n=MAX_POINT_ROWS, random_state=42)
        sampled = True
    subset.reset_index(drop=True, inplace=True)
    return subset, total, sampled


def build_color_ramp(
    series: pd.Series,
    percentile: float,
    start: tuple[int, int, int, int] = (255, 237, 160, 200),
    end: tuple[int, int, int, int] = (179, 0, 0, 230),
) -> list[list[int]]:
    """Create a colour ramp between two RGBA endpoints based on the series values."""
    values = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy()
    if values.size == 0:
        return []

    min_val = float(values.min())
    max_val = float(np.percentile(values, percentile))
    if min_val == max_val:
        return [list(end)] * len(values)

    clipped = np.clip(values, min_val, max_val)
    norm = (clipped - min_val) / (max_val - min_val)
    start_arr = np.array(start, dtype=float)
    end_arr = np.array(end, dtype=float)
    colors = (start_arr + (end_arr - start_arr) * norm[:, None]).astype(int)
    return colors.tolist()


def scale_radius(
    series: pd.Series,
    percentile: float,
    min_radius: float,
    max_radius: float,
) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").fillna(0).to_numpy()
    if values.size == 0:
        return np.array([])

    min_val = float(values.min())
    max_val = float(np.percentile(values, percentile))
    if min_val == max_val:
        return np.full(len(values), (min_radius + max_radius) / 2)

    clipped = np.clip(values, min_val, max_val)
    return np.interp(clipped, (min_val, max_val), (min_radius, max_radius))


def resolution_for_zoom(base_resolution: int, zoom: float) -> int:
    """Return an H3 resolution that adapts to the current zoom level."""
    if zoom <= 1.5:
        return max(MIN_H3_RESOLUTION, base_resolution - 2)
    if zoom <= 3:
        return max(MIN_H3_RESOLUTION, base_resolution - 1)
    if zoom >= 6:
        return min(MAX_H3_RESOLUTION, base_resolution + 1)
    return int(np.clip(base_resolution, MIN_H3_RESOLUTION, MAX_H3_RESOLUTION))


def append_location_info(
    df: pd.DataFrame,
    id_col: str = "h3_index",
    lat_col: str = "lat",
    lng_col: str = "lng",
) -> pd.DataFrame:
    """Add cached city/region/country fields for tooltip context."""
    if df.empty:
        df = df.copy()
        for col in ("city", "region", "country", "location_label"):
            df[col] = []
        return df

    df = df.copy()
    missing_keys = []
    query_coords = []

    for idx, key in df[id_col].items():
        if key not in LOCATION_CACHE:
            lat = float(df.at[idx, lat_col])
            lng = float(df.at[idx, lng_col])
            missing_keys.append((idx, key, lat, lng))
            query_coords.append((lat, lng))

    if query_coords:
        results = rg.search(query_coords, mode=1)
        for (idx, key, lat, lng), res in zip(missing_keys, results):
            country_obj = pycountry.countries.get(alpha_2=res.get("cc", ""))
            country_name = country_obj.name if country_obj else res.get("cc", "")
            city = res.get("name", "") or ""
            region = res.get("admin1", "") or ""
            parts = [part for part in [city, region, country_name] if part]
            label = ", ".join(parts) if parts else f"{lat:.2f}, {lng:.2f}"
            LOCATION_CACHE[key] = {
                "city": city,
                "region": region,
                "country": country_name,
                "location_label": label,
            }

    df["city"] = df[id_col].map(lambda key: LOCATION_CACHE.get(key, {}).get("city", ""))
    df["region"] = df[id_col].map(lambda key: LOCATION_CACHE.get(key, {}).get("region", ""))
    df["country"] = df[id_col].map(lambda key: LOCATION_CACHE.get(key, {}).get("country", ""))
    df["location_label"] = df[id_col].map(
        lambda key: LOCATION_CACHE.get(key, {}).get("location_label", "Unknown location")
    )
    return df


def build_h3_clusters(df: pd.DataFrame, resolution: int) -> pd.DataFrame:
    """Aggregate app opens into H3 clusters at the given resolution."""
    if df.empty:
        return pd.DataFrame(
            columns=
            [
                "h3_index",
                "hexIds",
                "total_opens",
                "unique_users",
                "mean_distance_km",
                "lat",
                "lng",
            ]
        )

    working = df.copy()
    working["distance_km"] = pd.to_numeric(working["Distance"], errors="coerce").fillna(0)
    working["h3_index"] = working.apply(
        lambda row: h3.latlng_to_cell(row["Lat"], row["Lng"], resolution), axis=1
    )

    grouped = (
        working.groupby("h3_index")
        .agg(
            total_opens=("Id", "count"),
            unique_users=("Id [User]", pd.Series.nunique),
            mean_distance=("distance_km", "mean"),
        )
        .reset_index()
    )

    grouped["mean_distance_km"] = grouped["mean_distance"].fillna(0).round(1)
    grouped.drop(columns=["mean_distance"], inplace=True)

    coords = grouped["h3_index"].apply(h3.cell_to_latlng)
    grouped["lat"] = coords.apply(lambda coord: float(coord[0]))
    grouped["lng"] = coords.apply(lambda coord: float(coord[1]))
    grouped["hexIds"] = grouped["h3_index"].apply(lambda idx: [idx])

    return grouped


def build_view_state(df: pd.DataFrame) -> pdk.ViewState:
    """Create a deck.gl view state centred on the data."""
    mean_lat = df["Lat"].mean()
    mean_lng = df["Lng"].mean()
    if pd.isna(mean_lat) or pd.isna(mean_lng):
        mean_lat, mean_lng = 0.0, 0.0

    zoom = 1.5
    if df["Lat"].std() < 10 and df["Lng"].std() < 10:
        zoom = 3

    return pdk.ViewState(latitude=float(mean_lat), longitude=float(mean_lng), zoom=zoom)


def render_map(df: pd.DataFrame) -> None:
    """Render destination-aware map views for the filtered dataset."""
    st.subheader("Map of App Opens")
    if df.empty:
        st.info("No rows match the selected date range.")
        return

    st.write(f"Showing {len(df):,} points spanning {df['Created at'].nunique()} days")

    min_distance = st.sidebar.slider(
        "Minimum distance (km) from nearest Published Tour",
        min_value=0,
        max_value=500,
        value=50,
        step=5,
    )
    st.sidebar.caption(
        "Rows below this threshold are filtered out. Increase to focus on remote listeners; decrease to include nearby users."
    )

    h3_resolution = st.sidebar.slider(
        "H3 resolution",
        min_value=4,
        max_value=9,
        value=DEFAULT_H3_RESOLUTION,
    )
    st.sidebar.caption("Higher values shrink hexagons for neighbourhood detail; lower values merge activity into broader regions.")
    color_percentile = st.sidebar.slider(
        "Cluster colour percentile",
        min_value=60,
        max_value=100,
        value=DEFAULT_COLOR_PERCENTILE,
    )
    st.sidebar.caption("Move right to reserve red for only the busiest cells; move left to colour more areas amber/red.")
    show_ports = st.sidebar.checkbox("Show world ports", value=True)
    st.sidebar.caption("Toggle to overlay global port locations for context.")
    max_radius = st.sidebar.slider(
        "Cluster max radius (m)",
        min_value=500,
        max_value=50000,
        value=DEFAULT_MAX_RADIUS,
        step=500,
    )
    st.sidebar.caption("Sets the largest bubble diameter when zoomed in. Increase for bold circles; decrease to keep them compact.")

    view_state = build_view_state(df)

    if min_distance > 0:
        df = df[pd.to_numeric(df["Distance"], errors="coerce").fillna(0) >= min_distance]
        if df.empty:
            st.warning("No rows match the selected distance threshold; relax the filter to see data.")
            return

    ports_df = load_ports() if show_ports else pd.DataFrame(columns=["latitude", "longitude", "port_label"])
    if show_ports and ports_df.empty:
        st.sidebar.info("Port dataset not available. Upload `world_ports.csv` to enable port pins.")

    adjusted_color_percentile = color_percentile
    adjusted_max_radius = max_radius
    if view_state.zoom <= GLOBAL_VIEW_ZOOM_THRESHOLD:
        adjusted_color_percentile = max(color_percentile, GLOBAL_COLOR_PERCENTILE)
        zoom_ratio = max(view_state.zoom / GLOBAL_VIEW_ZOOM_THRESHOLD, GLOBAL_RADIUS_SCALE)
        adjusted_max_radius = max(
            GLOBAL_MIN_RADIUS,
            min(max_radius, max_radius * zoom_ratio),
        )

    points = df.copy()
    points["route_title"] = points["Title [Route]"]
    points["user_id"] = points["Id [User]"]
    points["distance_km"] = pd.to_numeric(points["Distance"], errors="coerce").fillna(0).round(1)
    points["created_date"] = points["Created at"].dt.strftime("%Y-%m-%d")

    point_columns = ["Lat", "Lng", "route_title", "user_id", "distance_km", "created_date"]
    points_for_map, total_points, sampled = sample_points(points, point_columns)
    points_for_map["tooltip_html"] = points_for_map.apply(
        lambda row: (
            f"<b>{row['route_title']}</b><br/>Distance: {row['distance_km']} km"
            f"<br/>User ID: {row['user_id']}<br/>Date: {row['created_date']}"
        ),
        axis=1,
    )

    ports_layer = None
    if show_ports and not ports_df.empty:
        ports_overlay = ports_df.copy()
        ports_overlay["tooltip_html"] = ports_overlay["port_label"].apply(lambda name: f"<b>Port:</b> {name}")
        ports_layer = pdk.Layer(
            "ScatterplotLayer",
            data=ports_overlay,
            pickable=True,
            get_position="[longitude, latitude]",
            get_radius=18000,
            radius_scale=1,
            radius_min_pixels=4,
            radius_max_pixels=120,
            stroked=True,
            get_line_color=[255, 255, 255],
            line_width_min_pixels=1,
            get_fill_color=[0, 92, 230, 210],
        )

    point_layer = pdk.Layer(
        "ScatterplotLayer",
        data=points_for_map,
        pickable=True,
        get_position="[Lng, Lat]",
        get_radius=2200,
        radius_min_pixels=2,
        radius_max_pixels=18,
        stroked=False,
        get_fill_color=[218, 41, 28, 180],
    )

    point_layers = [point_layer]
    if ports_layer is not None:
        point_layers.append(ports_layer)

    point_tooltip = {
        "html": "{tooltip_html}",
        "style": {"color": "#111", "backgroundColor": "#f5f5f5"},
    }
    point_deck = pdk.Deck(
        layers=point_layers,
        initial_view_state=view_state,
        tooltip=point_tooltip,
        map_style=MAP_STYLE,
    )

    effective_resolution = resolution_for_zoom(h3_resolution, view_state.zoom)
    if effective_resolution != h3_resolution:
        st.caption(
            f"Using adaptive H3 resolution {effective_resolution} (base {h3_resolution}) for current zoom"
        )
    clusters = build_h3_clusters(df, effective_resolution)
    table_clusters = append_location_info(clusters)
    table_total = table_clusters.sort_values("total_opens", ascending=False)[
        ["location_label", "total_opens", "unique_users", "mean_distance_km"]
    ]
    table_unique = table_clusters.sort_values("unique_users", ascending=False)[
        ["location_label", "unique_users", "total_opens", "mean_distance_km"]
    ]

    cluster_tooltip = {
        "html": "{tooltip_html}",
        "style": {"color": "#111", "backgroundColor": "#fafafa"},
    }

    opens_deck = None
    users_deck = None
    if not clusters.empty:
        min_radius = max_radius * 0.25
        opens_colors = build_color_ramp(
            clusters["total_opens"], adjusted_color_percentile, start=(255, 247, 188, 150), end=(189, 0, 38, 240)
        )
        users_colors = build_color_ramp(
            clusters["unique_users"], adjusted_color_percentile, start=(237, 248, 233, 150), end=(0, 109, 44, 240)
        )
        radius_scaled = scale_radius(
            clusters["unique_users"], adjusted_color_percentile, min_radius=min_radius, max_radius=adjusted_max_radius
        )

        enriched = clusters.copy()
        enriched["fill_color_opens"] = opens_colors
        enriched["fill_color_users"] = users_colors
        enriched["radius_scaled"] = radius_scaled
        enriched = append_location_info(enriched)
        enriched["tooltip_html"] = enriched.apply(
            lambda row: (
                f"<b>{row['location_label']}</b><br/>"
                f"<b>Total opens:</b> {row['total_opens']:,}<br/>"
                f"<b>Unique users:</b> {row['unique_users']:,}<br/>"
                f"<b>Avg distance:</b> {row['mean_distance_km']} km"
            ),
            axis=1,
        )

        opens_data = enriched[
            [
                "hexIds",
                "total_opens",
                "unique_users",
                "mean_distance_km",
                "fill_color_opens",
                "radius_scaled",
                "location_label",
                "tooltip_html",
            ]
        ].copy()

        users_data = enriched[
            [
                "hexIds",
                "total_opens",
                "unique_users",
                "mean_distance_km",
                "fill_color_users",
                "radius_scaled",
                "location_label",
                "tooltip_html",
            ]
        ].copy()

        opens_points = enriched[["lng", "lat", "radius_scaled", "tooltip_html"]].copy()
        users_points = opens_points.copy()

        cluster_layers_opens = [
            pdk.Layer(
                "H3ClusterLayer",
                data=opens_data,
                pickable=True,
                stroked=True,
                filled=True,
                extruded=False,
                get_hexagons="hexIds",
                get_fill_color="fill_color_opens",
                get_line_color=[80, 80, 80],
                line_width_min_pixels=1,
            )
        ]
        cluster_layers_users = [
            pdk.Layer(
                "H3ClusterLayer",
                data=users_data,
                pickable=True,
                stroked=True,
                filled=True,
                extruded=False,
                get_hexagons="hexIds",
                get_fill_color="fill_color_users",
                get_line_color=[80, 80, 80],
                line_width_min_pixels=1,
            )
        ]

        if ports_layer is not None:
            cluster_layers_opens.append(ports_layer)
            cluster_layers_users.append(ports_layer)

        cluster_layers_opens.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=opens_points,
                get_position="[lng, lat]",
                get_radius="radius_scaled",
                radius_scale=1,
                radius_min_pixels=3,
                radius_max_pixels=100,
                stroked=False,
                get_fill_color=[255, 87, 34, 120],
                pickable=False,
            )
        )
        cluster_layers_users.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=users_points,
                get_position="[lng, lat]",
                get_radius="radius_scaled",
                radius_scale=1,
                radius_min_pixels=3,
                radius_max_pixels=100,
                stroked=False,
                get_fill_color=[0, 109, 44, 120],
                pickable=False,
            )
        )

        opens_deck = pdk.Deck(
            layers=cluster_layers_opens,
            initial_view_state=view_state,
            tooltip=cluster_tooltip,
            map_style=MAP_STYLE,
        )

        users_deck = pdk.Deck(
            layers=cluster_layers_users,
            initial_view_state=view_state,
            tooltip=cluster_tooltip,
            map_style=MAP_STYLE,
        )

    tab_points, tab_opens, tab_users = st.tabs(
        [
            "Individual opens",
            "H3 clusters by opens",
            "H3 clusters by unique users",
        ]
    )

    with tab_points:
        if sampled:
            st.caption(
                f"Rendering {len(points_for_map):,} of {total_points:,} app opens (cap {MAX_POINT_ROWS:,})."
            )
        st.pydeck_chart(point_deck, width="stretch", height=POINT_LAYER_HEIGHT)

    with tab_opens:
        if opens_deck is None:
            st.info("No H3 clusters were generated for the selected range.")
        else:
            st.pydeck_chart(opens_deck, width="stretch", height=CLUSTER_LAYER_HEIGHT)
            st.caption(
                "Top H3 cells by total opens — adjust colour percentile to tune yellow/red thresholds"
            )
            st.dataframe(table_total)

    with tab_users:
        if users_deck is None:
            st.info("No H3 clusters were generated for the selected range.")
        else:
            st.pydeck_chart(users_deck, width="stretch", height=CLUSTER_LAYER_HEIGHT)
            st.caption(
                "Top H3 cells by unique users — radius slider controls circle size"
            )
            st.dataframe(table_unique)

    with st.expander("Raw data sample"):
        st.dataframe(df.head(5000))
        st.caption(f"Showing first 5,000 of {len(df):,} rows. Export from the notebook for the full dataset.")


def main() -> None:
    st.set_page_config(page_title="VoiceMap App Opens", layout="wide")
    st.title("VoiceMap App Opens — Map View")
    st.caption("Filter app open coordinates by date and explore them on an interactive map.")

    data_path = discover_data_path()
    df = load_data(data_path)
    filtered = filter_by_date(df)
    render_map(filtered)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass
    except Exception as exc:  # pragma: no cover
        st.exception(exc)
        sys.exit(1)
