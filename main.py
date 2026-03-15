import tkinter as tk
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(__file__))

from app.main_window import MainWindow

if __name__ == "__main__":
    root = tk.Tk()
    app  = MainWindow(root)
    root.mainloop()
