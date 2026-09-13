"""Entry point with DPI-awareness enabled before Tk is created."""
from .hidpi import enable_dpi_awareness

enable_dpi_awareness()

from .editor import main

if __name__ == "__main__":
    main()
