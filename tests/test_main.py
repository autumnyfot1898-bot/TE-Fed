"""Method wrapper delegation to the CLI."""
import unittest
from unittest.mock import patch

from tefed.main import main, translate_argv


class MainEntryTests(unittest.TestCase):
    def test_method_is_forwarded_with_user_arguments(self):
        argv = ['tefed', '--config', 'config.toml', '--output', 'runs/run 1', '--seed', '4']
        with patch('tefed.main.cli.main', return_value=0) as downstream:
            self.assertEqual(main(argv), 0)
        downstream.assert_called_once_with(['train', '--method', 'tefed', *argv[1:]])

    def test_regular_commands_are_not_rewritten(self):
        for argv in [['smoke', '--output', 'runs/smoke'], ['evaluate', '--checkpoint', 'a.pt'], ['methods']]:
            with self.subTest(argv=argv):
                self.assertEqual(translate_argv(argv), argv)

    def test_conflicting_method_is_rejected(self):
        for flag in [['--method', 'fedavg'], ['--method=fedavg']]:
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ValueError, 'already selected'):
                    translate_argv(['tefed', *flag])

    def test_unknown_input_is_left_to_native_parser(self):
        self.assertEqual(translate_argv(['not-a-method']), ['not-a-method'])


if __name__ == '__main__':
    unittest.main()
