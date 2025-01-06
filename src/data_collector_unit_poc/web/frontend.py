"""NiceGUI frontend"""
import os
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from nicegui import app, ui
from nicegui.events import GenericEventArguments
from starlette.middleware.base import BaseHTTPMiddleware

from data_collector_unit_poc.data_storage import PostRepository


load_dotenv()

admin_user = os.getenv("ADMIN_USERNAME")
admin_pass = os.getenv("ADMIN_PASSWORD")

passwords = {admin_user: admin_pass}

unrestricted_page_routes = {'/login'}

class LazyLoaded(list):
    def __init__(self, *args, **kwargs):
        self.pagination = kwargs.pop('pagination', 1)
        super().__init__(*args, **kwargs)
        self()
    def __call__(self):
        for _ in range(self.pagination):
            self.append({"number": random.random()})
    @property
    def len(self):
        return len(self)

class PaginationChange:
    def __init__(self):
        self.maxpage = {}
    def __call__(self, e, data):
        prev = self.maxpage.get(str(e.sender), 0)
        curr = e.value['page']
        if curr > prev:
            data()
        e.sender.update()
        self.maxpage[str(e.sender)] = max(curr, self.maxpage.get(str(e.sender), 0))


def init(fastapi_app: FastAPI) -> None:
    
    repository = PostRepository()
    
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

        with ui.column().classes('absolute-center items-center'):
            ui.label(f'Hello {app.storage.user["username"]}!').classes('text-2xl')
            ui.button(on_click=logout, icon='logout').props('outline round')
            
    # # @ui.page('/')
    # @ui.page('/{_:path}')
    # def fallback(project: str = ''):
    #     if "username" not in app.storage.user:
    #         return RedirectResponse('/')

    @ui.page('/subpage')
    def test_page() -> None:
        ui.label('This is a sub page.')


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
    
    @ui.page("/posts")
    def posts() -> None:
        """Posts table"""
        
        columns = [
            {'name': 'title', 'label': 'Title', 'field': 'title'},
            {'name': 'link', 'label': 'URL', 'field': 'link'},
            {'name': 'timestamp', 'label': 'Timestamp', 'field': 'timestamp'},
        ]
        pagination = {'rowsPerPage': 10, 'rowsNumber': repository.count_all_posts(), 'page': 1}

        def get_rows():
            page = pagination['page']
            rpp = pagination['rowsPerPage']
            
            paged = repository.get_paginated_posts(page=page, page_size=rpp)
            
            return [{
                'title': post.title, 
                'link': post.url, 
                "timestamp": str(post.timestamp)
            } for post in paged]

        def on_request(e: GenericEventArguments) -> None:
            nonlocal pagination
            pagination = e.args['pagination']
            table.props('loading')
            rows = get_rows()
            table.pagination.update(pagination)
            table.props(remove='loading')
            table.update_rows(rows)

        table = ui.table(columns=columns, rows=get_rows(), row_key='name', pagination=pagination)
        table.add_slot('body', r'''
            <q-tr :props="props">
                <q-td v-for="col in props.cols" :key="col.name" :props="props">
                    <div v-if="col.name === 'link'" style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 40ch;">
                        <a :href="col.value" target="_blank">{{ col.value }}</a>
                    </div>
                    <div v-else style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 70ch;">
                        {{ col.value }}
                    </div>
                </q-td>
            </q-tr>
        ''')
        table.on('request', on_request)


    ui.run_with(
        fastapi_app,
        mount_path='/',  # NOTE this can be omitted if you want the paths passed to @ui.page to be at the root
        storage_secret=os.getenv("APP_SECRET"),
    )

