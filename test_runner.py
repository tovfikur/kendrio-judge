#!/usr/bin/env python3

import os
import sys
import unittest
import importlib.util
import glob
import io
import contextlib

# Add the current directory to the path so Python can find the modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def find_test_files():
    """Find all Python files in the current directory starting with 'Test', excluding the test runner."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_files = glob.glob(os.path.join(current_dir, "Test*.py"))
    runner_file = os.path.join(current_dir, "test_runner.py")
    return [f for f in test_files if os.path.isfile(f) and f != runner_file]

def load_test_module(test_file):
    """Load a test module from a file path."""
    module_name = os.path.splitext(os.path.basename(test_file))[0]
    spec = importlib.util.spec_from_file_location(module_name, test_file)
    if spec is None:
        print(f"Error: Could not create spec for {test_file}")
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as e:
        print(f"Error loading test module {test_file}: {str(e)}")
        return None

def main():
    """Main function to run all discovered tests."""
    print("Discovering test files in current directory...")
    test_files = find_test_files()

    if not test_files:
        print("Error: No test files found starting with 'Test' in the current directory.")
        sys.exit(1)

    print(f"Found {len(test_files)} test file(s): {', '.join(os.path.basename(f) for f in test_files)}")

    # Load all test modules and track loading errors
    test_modules = []
    load_errors = []
    for test_file in test_files:
        module = load_test_module(test_file)
        if module:
            test_modules.append(module)
        else:
            load_errors.append(test_file)

    if not test_modules and load_errors:
        print("Error: No valid test modules could be loaded.")
        print("Failed to load the following test files:")
        for err_file in load_errors:
            print(f"  - {os.path.basename(err_file)}")
        sys.exit(1)

    # Create a test suite from all modules
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for module in test_modules:
        module_suite = loader.loadTestsFromModule(module)
        suite.addTests(module_suite)

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        result = runner.run(suite)

    # Print the test results
    print(output.getvalue())

    # Summary and exit code
    if result.wasSuccessful() and not load_errors:
        print("All tests passed!")
        sys.exit(0)
    else:
        if load_errors:
            print("Errors occurred while loading test modules:")
            for err_file in load_errors:
                print(f"  - {os.path.basename(err_file)}: Failed to load module")
        if not result.wasSuccessful():
            print(f"Test failures: {len(result.failures)} failures, {len(result.errors)} errors")
        print("Test run completed with errors.")
        sys.exit(1)

if __name__ == "__main__":
    main()