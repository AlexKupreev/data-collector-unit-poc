"""Basic web UI

Login page + /admin page with restricted support.
"""
from nicegui import ui

# Global variable to track login state
is_logged_in = False

def authenticate(username, password):
    # Simple authentication logic (replace with real logic)
    return username == "admin" and password == "password"

def login():
    with ui.card():
        ui.label('Login')
        username = ui.input('Username')
        password = ui.input('Password', password=True)
        ui.button('Login', on_click=lambda: check_credentials(username.value, password.value))

def check_credentials(username, password):
    global is_logged_in
    if authenticate(username, password):
        is_logged_in = True
        ui.notify('Login successful!', color='green')
        ui.navigate.to('/admin')
    else:
        ui.notify('Invalid credentials', color='red')

@ui.page('/admin')
def admin_page():
    if not is_logged_in:
        ui.notify('Access denied. Please log in first.', color='red')
        ui.navigate.to('/')
    else:
        ui.label('Welcome to the Admin Page')
        ui.button('Logout', on_click=logout)

def logout():
    global is_logged_in
    is_logged_in = False
    ui.notify('Logged out successfully.', color='green')
    ui.navigate.to('/')

login()
ui.run()
