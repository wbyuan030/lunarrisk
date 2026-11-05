"""Synthetic native-grid policy tests, not official config/model execution."""
import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from lunarrisk.nasa_transfer.contracts import (
    prepare_terrain, validate_domain, spatial_policy_receipt, NATIVE_SPATIAL_POLICY,
    NASA_CONFIG_SHA256, MOONSTONE_NOMINAL_MPP,
)
from lunarrisk.nasa_transfer.dataset import TerrainDataset, NODATA_POLICY
from test_p33_nasa_contracts import fixture
from test_p33_nasa_dataset import RawSource


def native_fixture():
    raw, valid, domains, provenance = fixture()
    for domain in domains.values():
        domain.update(input_size=256, patch_size=16, pre_resize=512)
    raw = {key: np.zeros((2, 1, 256, 256), dtype=np.float32) for key in raw}
    raw['elevation'][:] = np.arange(256, dtype=np.float32)[None, None, None, :]
    raw['slope'][:] = 30
    valid = {key: np.ones_like(value, dtype=bool) for key, value in raw.items()}
    provenance.update(spatial_policy=NATIVE_SPATIAL_POLICY, meters_per_pixel=MOONSTONE_NOMINAL_MPP,
                      ground_sampling={'grid': 'equirectangular', 'pixels_per_degree': 128,
                                       'lunar_radius_m': 1737400, 'east_west_spacing': 'latitude_dependent',
                                       'nasa_native_60m_equivalence': False})
    return raw, valid, domains, provenance


def prepare(raw, valid, domains, provenance, **kwargs):
    return prepare_terrain(raw, valid, domains, provenance,
                           spatial_policy=NATIVE_SPATIAL_POLICY,
                           config_sha256=kwargs.get('config_sha256', NASA_CONFIG_SHA256))


class TestNativeSpatialPolicy(unittest.TestCase):
    def test_default_still_rejects_authentic_geometry(self):
        _, _, domains, _ = native_fixture()
        with self.assertRaisesRegex(ValueError, 'pre_resize differs'):
            validate_domain(domains['dtm'])

    def test_explicit_policy_preserves_grid_values_and_normalization(self):
        raw, valid, domains, provenance = native_fixture()
        before = {key: value.copy() for key, value in raw.items()}
        snapshot = copy.deepcopy(domains)
        prepared = prepare(raw, valid, domains, provenance)
        for source, destination in (('elevation', 'dtm'), ('slope', 'slope')):
            np.testing.assert_array_equal(prepared[destination], (raw[source] - 10) / 2)
            np.testing.assert_array_equal(raw[source], before[source])
        self.assertEqual(domains, snapshot)
        self.assertEqual(prepared['dtm'].shape, (2, 1, 256, 256))

    def test_native_requires_pinned_config_and_known_policy(self):
        for sha in (None, '', 'a' * 64):
            with self.assertRaisesRegex(ValueError, 'audited fixed-revision'):
                prepare(*native_fixture(), config_sha256=sha)
        with self.assertRaisesRegex(ValueError, 'unknown spatial policy'):
            spatial_policy_receipt('skip_guard', NASA_CONFIG_SHA256)

    def test_native_requires_declared_physical_scale_and_no_transform(self):
        for key, value in (('spatial_policy', None), ('ground_sampling', {}),
                           ('meters_per_pixel', 237.), ('meters_per_pixel', 60.),
                           ('spatial_transform', 'resize512_then_crop')):
            raw, valid, domains, provenance = native_fixture()
            provenance[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                prepare(raw, valid, domains, provenance)
        raw, valid, domains, provenance = native_fixture()
        provenance['ground_sampling']['nasa_native_60m_equivalence'] = True
        with self.assertRaises(ValueError):
            prepare(raw, valid, domains, provenance)

    def test_native_does_not_generalize_to_other_geometry(self):
        for key, value in (('input_size', 512), ('patch_size', 8), ('pre_resize', None),
                           ('pre_resize', 256), ('num_channels', 3)):
            raw, valid, domains, provenance = native_fixture()
            domains['dtm'][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                prepare(raw, valid, domains, provenance)
        raw, valid, domains, provenance = native_fixture()
        raw['elevation'] = raw['elevation'][:, :, :128]
        with self.assertRaisesRegex(ValueError, 'do not silently resize'):
            prepare(raw, valid, domains, provenance)

    def test_native_keeps_missingness_and_slope_guards(self):
        for defect in ('mask', 'nan', 'slope'):
            raw, valid, domains, provenance = native_fixture()
            if defect == 'mask':
                valid['elevation'][0, 0, 0, 0] = False
            elif defect == 'nan':
                raw['elevation'][0, 0, 0, 0] = np.nan
            else:
                raw['slope'][0, 0, 0, 0] = 91
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                prepare(raw, valid, domains, provenance)

    def test_native_dataset_threads_policy_without_normalized_reader(self):
        _, _, domains, provenance = native_fixture()
        provenance['nodata_policy'] = NODATA_POLICY
        source = RawSource()
        source.pixels = np.full((2, 28, 256, 256), 30., np.float32)
        dataset = TerrainDataset(source, {'a': 2, 'b': 3}, domains, provenance,
                                 spatial_policy=NATIVE_SPATIAL_POLICY, config_sha256=NASA_CONFIG_SHA256)
        np.testing.assert_array_equal(dataset[0]['dtm'], 10.)
        self.assertEqual(dataset[0]['sample_id'], 'a')
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]['dtm'].shape, (1, 256, 256))

    def test_protocol_matches_code_and_receipt_disclaims_compatibility(self):
        protocol = json.loads((Path(__file__).resolve().parents[1] /
                               'protocols/p33_native256_spatial_policy.json').read_text())
        receipt = spatial_policy_receipt(NATIVE_SPATIAL_POLICY, NASA_CONFIG_SHA256)
        self.assertEqual(protocol['policy_id'], receipt['policy_id'])
        self.assertEqual(protocol['nasa_config_sha256'], receipt['config_sha256'])
        self.assertEqual(protocol['model_geometry']['input_size'], receipt['model_shape'])
        self.assertFalse(receipt['nasa_native_scale_equivalence'])
        self.assertEqual(receipt['positional_embedding_compatibility'],
                         'REQUIRES_STRICT_CHECKPOINT_AND_REAL_PREFLIGHT')


if __name__ == '__main__':
    unittest.main()
