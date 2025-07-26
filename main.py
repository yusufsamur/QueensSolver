from gui import QueensGameSolver
import tkinter as tk

def main():
    root = tk.Tk()
    app = QueensGameSolver(root)
    root.mainloop()

if __name__ == "__main__":
    main()