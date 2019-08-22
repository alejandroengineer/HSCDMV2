import glfw

glfw.init()

glfw.window_hint(glfw.CENTER_CURSOR, glfw.FALSE)

window = glfw.create_window(640, 480, "Window Hint Bug", glfw.get_primary_monitor(), None)