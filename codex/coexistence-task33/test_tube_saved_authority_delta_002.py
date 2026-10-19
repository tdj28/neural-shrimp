"""Only the corrected independent-input-pin negative control; no old suite run."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('authority_original_fixture', Path(__file__).with_name('test_tube_saved_authority.py'))
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


class CorrectedInputPinControl(unittest.TestCase):
    def test_independent_card_input_mismatch_is_refused(self):
        files, execution, assets, scalar = prior.M.load_helpers()
        io, authority, values = prior.upstream_fixture(execution)
        card = values[authority['card']['path']]
        plan = values[authority['plan']['path']]
        card['inputs'] = copy.deepcopy(card['inputs'])
        planned = plan['lanes']['synthetic']['preflight_pins'][0]
        self.assertIsNot(card['inputs'][0], planned)
        unchanged = copy.deepcopy(planned)
        card['inputs'][0]['sha256'] = '0' * 64
        self.assertEqual(planned, unchanged)
        self.assertNotEqual(card['inputs'][0], planned)
        with self.assertRaisesRegex(ValueError, 'upstream declared inputs absent from preflight'):
            prior.M.original_execution(io, authority)
        io.verify_pins.assert_not_called()
        io.inventory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
