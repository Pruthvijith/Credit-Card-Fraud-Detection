from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from fraud_platform.registry import checked_release, promote, register


class RegistryTests(TestCase):
    def test_repeat_register_rollback_and_reject_corrupt_candidate(self):
        root = Path(self.enterContext(TemporaryDirectory()))
        first = register(root, 1200, 17)
        self.assertEqual(register(root, 1200, 17), first)
        second = register(root, 1200, 21)
        self.assertNotEqual(first, second)
        for version in [first, second, first]:
            promote(root, version)
            self.assertEqual((root / 'active').read_text().strip(), version)
        (checked_release(root, second) / 'model.joblib').write_bytes(b'damaged')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            promote(root, second)
        self.assertEqual((root / 'active').read_text().strip(), first)
        with self.assertRaises(ValueError):
            checked_release(root, '../other')
