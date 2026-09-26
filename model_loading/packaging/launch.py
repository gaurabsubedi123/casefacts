"""Entry point for the frozen .exe.

PyInstaller runs its entry script as a top-level file, where the relative
imports inside modelportal/__main__.py would fail, so this imports the
package properly and hands over.
"""

import sys

from modelportal.__main__ import main

if __name__ == "__main__":
    sys.exit(main())
