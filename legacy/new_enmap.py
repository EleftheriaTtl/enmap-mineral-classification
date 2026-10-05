import warnings
warnings.filterwarnings("ignore")

# Ensure PROJ uses a compatible database before rasterio/rioxarray imports
import os
from pathlib import Path
import importlib.util

os.environ.pop("PROJ_LIB", None)
os.environ.pop("PROJ_DATA", None)
spec = importlib.util.find_spec("rasterio")
if spec and spec.origin:
    rasterio_dir = Path(spec.origin).resolve().parent
    rasterio_proj = rasterio_dir / "proj_data"
    if (rasterio_proj / "proj.db").exists():
        os.environ["PROJ_DATA"] = str(rasterio_proj)

import numpy as np
import xarray as xr
import rioxarray
import xml.etree.ElementTree as ET
from collections import OrderedDict

import zipfile
import tarfile
import tempfile
from typing import Optional, List, Dict


class EnMAPProcessor:
    def __init__(
        self,
        base_dir: str,
        save_netcdf: bool = False,
        netcdf_dir: Optional[str] = None,
        load_quality_layers: bool = False,
        apply_cloud_mask: bool = False,
        remove_clouds: bool = True,
        remove_cloud_shadow: bool = True,
        remove_haze: bool = True,
        remove_snow: bool = True,
        remove_cirrus: bool = True,
        remove_pixel_mask: bool = True,
        remove_classes: bool = True,
        # spectral options
        keep_only_swir: bool = True,
        version: str = "v1",
    ):
        self.base_dir = Path(base_dir)

        self.save_netcdf = save_netcdf
        self.netcdf_dir = (self.base_dir / "netcdf_outputs") if netcdf_dir is None else Path(netcdf_dir)
        if self.save_netcdf:
            self.netcdf_dir.mkdir(parents=True, exist_ok=True)

        self.load_quality_layers = load_quality_layers
        self.apply_cloud_mask = apply_cloud_mask

        self.remove_clouds = remove_clouds
        self.remove_cloud_shadow = remove_cloud_shadow
        self.remove_haze = remove_haze
        self.remove_snow = remove_snow
        self.remove_cirrus = remove_cirrus
        self.remove_pixel_mask = remove_pixel_mask
        self.remove_classes = remove_classes
        self.keep_only_swir = keep_only_swir
        self.version = version
    # ---------------------------------------------------------------------
    # Discovery
    # ---------------------------------------------------------------------
    def find_enmap_data_sources(self) -> Dict[str, List[Path]]:
        sources = {"tarballs": [], "extracted_dirs": [], "zip_files": []}
        # Exclude macOS AppleDouble files (._*.tar.gz), which are not real gzip
        sources["tarballs"] = [p for p in self.base_dir.glob("dims*.tar.gz") if not p.name.startswith("._")]

        for item in self.base_dir.iterdir():
            if item.is_dir():
                l2a_dir = item / "ENMAP.HSI.L2A"
                if l2a_dir.exists():
                    sources["extracted_dirs"].append(item)

        sources["zip_files"] = list(self.base_dir.rglob("*.ZIP")) + list(self.base_dir.rglob("*.zip"))
        return sources

    # ---------------------------------------------------------------------
    # Main read
    # ---------------------------------------------------------------------
    def read_enmap_zip(self, zip_path: Path) -> Optional[xr.Dataset]:
        print(f"\nReading: {zip_path.name}")
        try:
            if zipfile.is_zipfile(zip_path):
                with tempfile.TemporaryDirectory(prefix="enmap_temp_") as tmpdir:
                    with zipfile.ZipFile(zip_path, "r") as z:
                        z.extractall(tmpdir)
                    return self._process_enmap_files(Path(tmpdir), zip_path)
            else:
                return self._process_enmap_files(zip_path.parent, zip_path)
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_enmap_files(self, dir_path: Path, source_path: Path) -> Optional[xr.Dataset]:
        spectral_files = list(dir_path.rglob("*SPECTRAL_IMAGE*.TIF")) + list(dir_path.rglob("*SPECTRAL_IMAGE*.tif"))
        if not spectral_files:
            print("  ⚠ No SPECTRAL_IMAGE file found")
            return None

        spectral_file = spectral_files[0]
        print(f"  Found: {spectral_file.name}")

        da = rioxarray.open_rasterio(spectral_file, masked=True)  # (band,y,x)
        n_bands, height, width = da.shape

        # Parse true wavelengths + detector identity from XML (no fake linspace)
        wl_all, vnir, swir, wl_center, fwhm, swir_idx, xml_used = self._parse_metadata_wavelengths(dir_path)



        if len(wl_all) != n_bands:
            raise ValueError(f"Band count mismatch: raster={n_bands}, XML={len(wl_all)} ({xml_used.name})")

        data = da.values.astype(np.float32)  # (band,y,x) native order
        data_full = data  # keep full-spectrum copy for index calculations

        # BSI/MBI (compute before any spectral filtering; uses VNIR+SWIR bands)
        bsi_blue_nm = 490.0
        bsi_red_nm = 665.0
        bsi_nir_nm = 850.0
        bsi_swir1_nm = 1610.0
        bsi_tol_nm = 20.0

        mbi_swir1_nm = 1610.0
        mbi_swir2_nm = 2200.0
        mbi_nir_nm = 850.0
        mbi_tol_nm = 20.0

        wl_full = np.asarray(wl_center, dtype=np.float32)
        blue_idx = int(np.argmin(np.abs(wl_full - bsi_blue_nm)))
        red_idx = int(np.argmin(np.abs(wl_full - bsi_red_nm)))
        nir_idx = int(np.argmin(np.abs(wl_full - bsi_nir_nm)))
        swir1_idx = int(np.argmin(np.abs(wl_full - bsi_swir1_nm)))
        swir2_idx = int(np.argmin(np.abs(wl_full - mbi_swir2_nm)))

        blue_nm = float(wl_full[blue_idx])
        red_nm = float(wl_full[red_idx])
        nir_nm = float(wl_full[nir_idx])
        swir1_nm = float(wl_full[swir1_idx])
        swir2_nm = float(wl_full[swir2_idx])


        # -------------------------------------------------------
        # FULL SPECTRUM MODE (VNIR + SWIR): do NOT slice by swir_idx
        # -------------------------------------------------------
        if self.keep_only_swir:
            data = data[swir_idx, :, :]             
            wl_center = wl_center[swir_idx]
            fwhm = fwhm[swir_idx]

        # Sort ALL bands by wavelength (monotonic axis)
        wl_full = np.asarray(wl_center, dtype=np.float32)
        fwhm_full = np.asarray(fwhm, dtype=np.float32)
        sort_idx = np.argsort(wl_full)

        wl_sorted = wl_full[sort_idx]
        data_sorted = data[sort_idx, :, :].astype(np.float32)
        fwhm_sorted = fwhm_full[sort_idx]

        # Build ds
        data_vars = {
            "reflectance": (
                ["wavelength", "y", "x"],
                data_sorted,
                {
                    "long_name": "Surface Reflectance",
                    "units": "reflectance",
                    'scale_factor': 0.0001,
                    "nodata": da.rio.nodata if da.rio.nodata is not None else -999,
                },
            )
        }


        if (
            abs(blue_nm - bsi_blue_nm) <= bsi_tol_nm
            and abs(red_nm - bsi_red_nm) <= bsi_tol_nm
            and abs(nir_nm - bsi_nir_nm) <= bsi_tol_nm
            and abs(swir1_nm - bsi_swir1_nm) <= bsi_tol_nm
        ):
            blue = data_full[blue_idx, :, :]
            red = data_full[red_idx, :, :]
            nir = data_full[nir_idx, :, :]
            swir1 = data_full[swir1_idx, :, :]
            bsi = ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue) + 1e-12)
            data_vars["bsi"] = (
                ["y", "x"],
                bsi.astype(np.float32),
                {
                    "long_name": "Bare Soil Index",
                    "blue_nm": blue_nm,
                    "red_nm": red_nm,
                    "nir_nm": nir_nm,
                    "swir1_nm": swir1_nm,
                },
            )
        else:
            print("  ⚠ BSI skipped: missing bands near 490/665/850/1610 nm")

        if (
            abs(swir1_nm - mbi_swir1_nm) <= mbi_tol_nm
            and abs(swir2_nm - mbi_swir2_nm) <= mbi_tol_nm
            and abs(nir_nm - mbi_nir_nm) <= mbi_tol_nm
        ):
            swir1 = data_full[swir1_idx, :, :]
            swir2 = data_full[swir2_idx, :, :]
            nir = data_full[nir_idx, :, :]
            mbi = ((swir1 - swir2 - nir) / (swir1 + swir2 + nir + 1e-12)) + 0.5
            data_vars["mbi"] = (
                ["y", "x"],
                mbi.astype(np.float32),
                {
                    "long_name": "Modified Bare Soil Index",
                    "nir_nm": nir_nm,
                    "swir1_nm": swir1_nm,
                    "swir2_nm": swir2_nm,
                },
            )
        else:
            print("  ⚠ MBI skipped: missing bands near 850/1610/2200 nm")

        quality_data = {}
        if self.load_quality_layers or self.apply_cloud_mask:
            quality_data = self._load_quality_layers(spectral_file.parent, da)
            if self.load_quality_layers:
                for layer_name, layer_array in quality_data.items():
                    data_vars[layer_name] = (
                        ["y", "x"],
                        layer_array.astype(np.uint8),
                        {"long_name": f"Quality layer: {layer_name}", "description": self._get_quality_description(layer_name)},
                    )
        # Build detector labels aligned to wl_center: VNIR if in vnir list else SWIR if in swir list
        # Use tolerance matching (floats won't match exactly).
        vnir_arr = np.asarray(vnir, dtype=np.float32)
        swir_arr = np.asarray(swir, dtype=np.float32)
        tol_nm = 0.6

        ds = xr.Dataset(
            data_vars=data_vars,
            coords={
                "wavelength": ("wavelength", wl_sorted, {"units": "nm"}),
                "y": da.y,
                "x": da.x,
            },
            attrs={
                "crs": str(da.rio.crs) if da.rio.crs else None,
                "crs_wkt": da.rio.crs.to_wkt() if da.rio.crs else None,
                "transform": list(da.rio.transform())[:6],
                "bounds": da.rio.bounds(),
                "source_file": source_path.name,
                "sensor": "EnMAP",
                "processing_level": "L2A",
                "metadata_xml": xml_used.name,
                "n_bands_raw": int(n_bands),
                "height": int(height),
                "width": int(width),
            },
        )

        # Ensure rioxarray sees CRS + spatial dims (prevents MissingCRS on clip)
        if da.rio.crs is None:
            raise ValueError("GeoTIFF has no CRS. Cannot georeference dataset.")
        ds = ds.rio.write_crs(da.rio.crs)
        ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y")

        ds, wl_sorted, fwhm_sorted = self.drop_bands(ds, wl_sorted, fwhm_sorted)
        # ---------------------------------------------------------------------
        # Save SRF in pkl using wl_swir and fwhm_swir
        # ---------------------------------------------------------------------
        import pickle

        srf_dict = OrderedDict(zip(wl_sorted, fwhm_sorted))
        srf_pkl_path = self.base_dir / "full_srf.pkl"
        with open(srf_pkl_path, "wb") as f:
            pickle.dump(srf_dict, f)

        # Optional quality mask (pixel-wise)
        if self.apply_cloud_mask and quality_data:
            mask = self._create_quality_mask(quality_data)
            if mask is not None:
                ds = self._apply_mask_to_dataset(ds, mask)

        print(f"  ✓ Loaded: {ds.sizes['wavelength']} bands (SWIR-only={self.keep_only_swir}), {height}x{width}")
        print(f"  ✓ Wavelength range: {float(ds.wavelength.min()):.1f} - {float(ds.wavelength.max()):.1f} nm")

        if self.save_netcdf:
            self._save_as_netcdf(ds, source_path)

        return ds

    # ---------------------------------------------------------------------
    # XML parsing
    # ---------------------------------------------------------------------
    import numpy as np
    import xml.etree.ElementTree as ET

    def _parse_metadata_wavelengths(self, dir_path: Path):
        import re
        from collections import defaultdict

        # Different EnMAP distributors/package formats use different metadata filenames:
        # - EOWEB:    ENMAP01-...-METADATA.XML
        # - GeoServer: METADATA.XML (no prefix)
        xml_files = (
            list(dir_path.rglob("*-METADATA.XML"))
            + list(dir_path.rglob("*-METADATA.xml"))
            + list(dir_path.rglob("METADATA.XML"))
            + list(dir_path.rglob("METADATA.xml"))
        )
        if not xml_files:
            raise FileNotFoundError(
                "No METADATA XML found. Expected one of: *-METADATA.XML, METADATA.XML "
                "(no fabricated wavelengths allowed)."
            )

        xml_path = xml_files[0]

        # ---- your existing agg-based extraction (keep it) ----
        float_re = re.compile(r"[-+]?\d*\.\d+|\d+")
        def floats_from_string(s):
            if not s:
                return []
            vals = [float(v) for v in float_re.findall(s)]
            return [v for v in vals if 350 <= v <= 2550]

        path_stack = []
        agg = defaultdict(list)
        for event, elem in ET.iterparse(xml_path, events=("start", "end")):
            if event == "start":
                tag = elem.tag.split("}")[-1]
                path_stack.append(tag)
            else:
                path = "/".join(path_stack)
                agg[path].extend(floats_from_string((elem.text or "").strip()))
                for _, v in elem.attrib.items():
                    agg[path].extend(floats_from_string(str(v)))
                path_stack.pop()
                elem.clear()

        wl_all_path = "level_X/specific/bandCharacterisation/bandID/wavelengthCenterOfBand"
        vnir_path   = "level_X/product/smileCorrection/VNIR/bandID/wavelength"
        swir_path   = "level_X/product/smileCorrection/SWIR/bandID/wavelength"

        wl_all = np.array(agg.get(wl_all_path, []), dtype=float)
        vnir   = np.array(agg.get(vnir_path, []), dtype=float)
        swir   = np.array(agg.get(swir_path, []), dtype=float)

        if len(wl_all) == 0 or len(swir) == 0:
            raise ValueError(f"Failed wavelength parse from {xml_path.name}: wl_all={len(wl_all)}, swir={len(swir)}")

        # ------------------------------------------------------------------
        # NEW: extract FWHM aligned to wl_all using the same XML
        # ------------------------------------------------------------------
        # Parse again (simple) to get bandCharacterisation lists in order
        tree = ET.parse(xml_path)
        root = tree.getroot()

        def strip_ns(tag):  # namespace-safe
            return tag.split("}")[-1]

        wl_bc = []
        fwhm_bc = []

        for band in root.iter():
            if strip_ns(band.tag) != "bandID":
                continue

            # only keep bandID elements that are under bandCharacterisation
            # quick filter: check ancestors is hard in ElementTree; instead check children presence
            w = None
            fw = None
            for child in band:
                t = strip_ns(child.tag)
                if t == "wavelengthCenterOfBand" and child.text:
                    w = float(child.text)
                elif t in {"fwhmOfBand", "FWHMOfBand", "FWHM"} and child.text:
                    fw = float(child.text)

            if w is not None:
                wl_bc.append(w)
                fwhm_bc.append(fw if fw is not None else np.nan)

        wl_bc = np.array(wl_bc, dtype=np.float32)
        fwhm_bc = np.array(fwhm_bc, dtype=np.float32)

        # Sanity: wl_bc should match wl_all length (usually 224)
        if len(wl_bc) != len(wl_all):
            # fall back: you can still proceed, but warn loudly
            print(f"⚠ bandCharacterisation wl length {len(wl_bc)} != wl_all length {len(wl_all)} (XML quirks).")

        # ------------------------------------------------------------------
        # NEW: SWIR indices via tolerance matching (NOT exact float isin)
        # ------------------------------------------------------------------
        # map each swir wavelength to nearest center wavelength
        tol_nm = 0.6  # conservative
        swir_idx = []
        for w in swir.astype(np.float32):
            j = int(np.argmin(np.abs(wl_bc - w)))
            if abs(wl_bc[j] - w) > tol_nm:
                raise ValueError(f"SWIR wavelength {w:.2f} nm not matched to center list (nearest {wl_bc[j]:.2f}).")
            swir_idx.append(j)

        swir_idx = np.array(sorted(set(swir_idx)), dtype=int)

        # Return wl_all etc plus fwhm + swir indices (you choose what to return)
        return wl_all, vnir, swir, wl_bc, fwhm_bc, swir_idx, xml_path

    # ---------------------------------------------------------------------
    # Quality layers + mask (FIXED LOGIC)
    # ---------------------------------------------------------------------
    def _load_quality_layers(self, qdir: Path, reference_da) -> Dict[str, np.ndarray]:
        quality_patterns = {
            "classes": "*QL_QUALITY_CLASSES.TIF",
            "cloud": "*QL_QUALITY_CLOUD.TIF",
            "cloud_shadow": "*QL_QUALITY_CLOUDSHADOW.TIF",
            "haze": "*QL_QUALITY_HAZE.TIF",
            "cirrus": "*QL_QUALITY_CIRRUS.TIF",
            "snow": "*QL_QUALITY_SNOW.TIF",
            "pixel_mask": "*QL_PIXELMASK.TIF",
        }

        out: Dict[str, np.ndarray] = {}
        for name, pattern in quality_patterns.items():
            files = list(qdir.glob(pattern))
            if not files:
                continue
            layer_da = rioxarray.open_rasterio(files[0])

            if (
                (layer_da.rio.crs != reference_da.rio.crs)
                or (layer_da.rio.transform() != reference_da.rio.transform())
                or (layer_da.sizes.get("y") != reference_da.sizes["y"])
                or (layer_da.sizes.get("x") != reference_da.sizes["x"])
            ):
                layer_da = layer_da.rio.reproject_match(reference_da, resampling="nearest")

            arr = layer_da.values
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
            elif arr.ndim == 3:
                arr = (arr != 0).any(axis=0).astype(np.uint8)

            nodata = layer_da.rio.nodata
            if nodata is not None:
                arr = np.where(arr == nodata, 0, arr)

            out[name] = arr.astype(np.uint8)
            print(f"  ✓ Loaded {name} layer")

        return out

    def _get_quality_description(self, layer_name: str) -> str:
        return {
            "cloud": "0=Clear, 1=Cloud",
            "cloud_shadow": "0=Clear, 1=Shadow",
            "haze": "0=Clear, 1=Haze",
            "snow": "0=Clear, 1=Snow",
            "cirrus": "0=Clear, 1=Cirrus",
            "pixel_mask": "0=Clear, 1=Flagged",
            "classes": "0=Clear, >0=Flagged",
        }.get(layer_name, "Quality layer")

    def _create_quality_mask(self, quality_data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        if not quality_data:
            return None

        # True = keep
        sample = next(iter(quality_data.values()))
        H, W = sample.shape
        mask = np.ones((H, W), dtype=bool)

        if self.remove_clouds and "cloud" in quality_data:
            mask &= (quality_data["cloud"] == 0)
        if self.remove_cloud_shadow and "cloud_shadow" in quality_data:
            mask &= (quality_data["cloud_shadow"] == 0)
        if self.remove_haze and "haze" in quality_data:
            mask &= (quality_data["haze"] == 0)
        if self.remove_snow and "snow" in quality_data:
            mask &= (quality_data["snow"] == 0)
        if self.remove_cirrus and "cirrus" in quality_data:
            mask &= (quality_data["cirrus"] == 0)

        # FIXED: keep clear
        if self.remove_pixel_mask and "pixel_mask" in quality_data:
            mask &= (quality_data["pixel_mask"] == 1)

        # FIXED: keep clear
        if self.remove_classes and "classes" in quality_data:
            mask &= (quality_data["classes"] == 1)

        return mask

    def _apply_mask_to_dataset(self, ds: xr.Dataset, mask: np.ndarray) -> xr.Dataset:
        mask_da = xr.DataArray(mask, dims=("y", "x"), coords={"y": ds.y, "x": ds.x})
        ds["reflectance"] = ds["reflectance"].where(mask_da)
        return ds

    # ---------------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------------
    def _save_as_netcdf(self, ds: xr.Dataset, source_path: Path) -> None:
        filename = source_path.stem
        parts = filename.split("_")
        product_id = "_".join(parts[:-2]) if len(parts) >= 3 else filename
        out = self.netcdf_dir / f"{product_id}.nc"

        encoding = {
            "reflectance": {
                "zlib": True,
                "complevel": 4,
                "dtype": "float32",
                "chunksizes": (min(10, ds.sizes["wavelength"]), min(256, ds.sizes["y"]), min(256, ds.sizes["x"])),
            }
        }
        ds.to_netcdf(out, engine="netcdf4", encoding=encoding)
        print(f"  ✓ Saved: {out.name}")

    # ---------------------------------------------------------------------
    # Batch processing (unchanged)
    # ---------------------------------------------------------------------
    def process_extracted_directory(self, dir_path: Path) -> List[xr.Dataset]:
        l2a_dir = dir_path / "ENMAP.HSI.L2A"
        if not l2a_dir.exists():
            print("  ⚠ No ENMAP.HSI.L2A directory found")
            return []
        zip_files = sorted(l2a_dir.glob("*.ZIP"))
        datasets = []
        for z in zip_files:
            ds = self.read_enmap_zip(z)
            if ds is not None:
                datasets.append(ds)
        return datasets

    def process_all_available_data(self) -> Dict[str, List[xr.Dataset]]:
        sources = self.find_enmap_data_sources()
        all_datasets: Dict[str, List[xr.Dataset]] = {}

        for extracted_dir in sorted(sources["extracted_dirs"]):
            dsets = self.process_extracted_directory(extracted_dir)
            if dsets:
                all_datasets[extracted_dir.name] = dsets

        for tarball in sorted(sources["tarballs"]):
            extracted_name = tarball.stem.replace(".tar", "")
            if extracted_name in [d.name for d in sources["extracted_dirs"]]:
                continue
            # Detect format: gzip magic \x1f\x8b; otherwise try uncompressed tar
            try:
                with open(tarball, "rb") as f:
                    magic = f.read(2)
                mode = "r:gz" if magic == b"\x1f\x8b" else "r"
            except Exception:
                mode = "r:gz"
            try:
                with tarfile.open(tarball, mode) as tar:
                    tar.extractall(path=self.base_dir)
                    members = tar.getmembers()
                    if members:
                        root_dir = Path(members[0].name).parts[0]
                        extracted_path = self.base_dir / root_dir
                        dsets = self.process_extracted_directory(extracted_path)
                        if dsets:
                            all_datasets[extracted_path.name] = dsets
            except tarfile.ReadError as e:
                print(f"  ⚠ Skipping {tarball.name}: not a valid tar/gzip file ({e})")

        for zip_file in sorted(sources["zip_files"]):
            ds = self.read_enmap_zip(zip_file)
            if ds is not None:
                all_datasets.setdefault(zip_file.parent.name, []).append(ds)

        return all_datasets

    def drop_bands(
        self,
        ds: xr.Dataset,
        wavelength: Optional[np.ndarray] = None,
        fwhm: Optional[np.ndarray] = None,
    ) -> tuple[xr.Dataset, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Drop known bad bands for 224-band EnMAP products.
        Also remove the same indices from wavelength and fwhm if provided.
        """
        w = ds["wavelength"].values
        if self.version == "128":
            drop_mask = ((w < 1449.43) & (w > 1330.86))
        elif self.version == "58":
            drop_mask = (w < 1967.66)
        else:
            drop_mask = ~np.ones_like(w, dtype=bool)
        drop_idx = np.where(drop_mask)[0].tolist()
        if drop_idx:
            ds = ds.drop_isel(wavelength=drop_idx)
            if wavelength is not None and len(wavelength) == len(w):
                wavelength = np.delete(wavelength, drop_idx)
            if fwhm is not None and len(fwhm) == len(w):
                fwhm = np.delete(fwhm, drop_idx)
        return ds, wavelength, fwhm
        