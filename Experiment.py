import numpy as np
from tkinter import Tk, Label, Button, messagebox, OptionMenu, StringVar
import pyglet
import SLM
import PCAM

event_loop = pyglet.app.EventLoop()

@event_loop.event
def on_window_close(window):
    event_loop.exit()

root = Tk()

screens = SLM.list_screens()

tkvar = StringVar(root)
tkvar.set(screens[0])
option = OptionMenu(root, tkvar, set(screens))
option.pack()

tkvar2 = StringVar(root)
tkvar2.set(screens[0])
option2 = OptionMenu(root, tkvar2, set(screens))
option2.pack()

def start():
    print('starting experiment')
    SLM1 = SLM.SLM(0)
    

start_button = Button(root, text="Start", command=start)
start_button.pack()


def on_closing():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        event_loop.exit()
        root.destroy()

root.protocol("WM_DELETE_WINDOW", on_closing)

root.mainloop()