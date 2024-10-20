"""NiceGUI frontend"""
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from nicegui import app, ui
from nicegui.events import GenericEventArguments
from starlette.middleware.base import BaseHTTPMiddleware

from data_collector_unit_poc.web.scheduler import build_now_trigger, ingest_noaa_isd_lite_job, scheduler


load_dotenv()

admin_user = os.getenv("ADMIN_USERNAME")
admin_pass = os.getenv("ADMIN_PASSWORD")

passwords = {admin_user: admin_pass}

unrestricted_page_routes = {'/login'}

def init(fastapi_app: FastAPI) -> None:
    
    class AuthMiddleware(BaseHTTPMiddleware):
        """This middleware restricts access to all NiceGUI pages.

        It redirects the user to the login page if they are not authenticated.
        """

        async def dispatch(self, request: Request, call_next):
            if not app.storage.user.get('authenticated', False):
                if not request.url.path.startswith('/_nicegui') and request.url.path not in unrestricted_page_routes:
                    app.storage.user['referrer_path'] = request.url.path  # remember where the user wanted to go
                    return RedirectResponse('/login')
            return await call_next(request)
        
    app.add_middleware(AuthMiddleware)
    
    @ui.page('/')
    def main_page() -> None:
        def logout() -> None:
            app.storage.user.clear()
            ui.navigate.to('/login')

        with ui.grid(columns=2):
            ui.button('Ingest NOAA ISD', on_click=lambda: (scheduler.add_job(ingest_noaa_isd_lite_job, build_now_trigger()), ui.notify('Ingest NOAA ISD job triggered successfully!'))).props('outline round')
            
            ui.label(f'Hello {app.storage.user["username"]}!').classes('text-2xl')
            ui.button(on_click=logout, icon='logout').props('outline round')

    @ui.page('/login')
    def login() -> Optional[RedirectResponse]:
        def try_login() -> None:  # local function to avoid passing username and password as arguments
            if passwords.get(username.value) == password.value:
                app.storage.user.update({'username': username.value, 'authenticated': True})
                ui.navigate.to(app.storage.user.get('referrer_path', '/'))  # go back to where the user wanted to go
            else:
                ui.notify('Wrong username or password', color='negative')

        if app.storage.user.get('authenticated', False):
            return RedirectResponse('/')
        with ui.card().classes('absolute-center'):
            username = ui.input('Username').on('keydown.enter', try_login)
            password = ui.input('Password', password=True, password_toggle_button=True).on('keydown.enter', try_login)
            ui.button('Log in', on_click=try_login)
        return None

    ui.run_with(
        fastapi_app,
        mount_path='/',  # NOTE this can be omitted if you want the paths passed to @ui.page to be at the root
        storage_secret=os.getenv("APP_SECRET"),
    )
