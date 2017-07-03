import React from 'react'
import {render} from 'react-dom'
import {Route, BrowserRouter} from 'react-router-dom'

import {Header} from './header.jsx'
import {Deadman} from './framework/deadman.jsx'
import {Index} from './modules/index.jsx'
import {Drive} from './modules/drive.jsx'
import {Test} from './modules/test.jsx'
import {SHM} from './modules/shm.jsx'
import {Admin} from './modules/admin.jsx'

const blueprint = [
    {
        name: "Index",
        path: "/",
        component: Index,
    },
    {
        name: "Drive",
        path: "/drive",
        component: Drive
    },
    {
        name: "Test",
        path: "/test",
        component: Test
    },
    {
        name: "SHM",
        path: "/shm",
        component: SHM
    },
    {
        name: "Admin",
        path: "/admin",
        component: Admin
    },
];

function generateRoutes() {
    return (
        blueprint.map(component =>
            // we want "exact" matching for the url === location.pathname
            <Route exact key={component.name} path={component.path} component={component.component} />
        )
    );
}

class Layout extends React.Component {
    render() {
        return (
            <div>
                <Header links={blueprint} />
                <div class="container">
                    {generateRoutes()}
                </div>
                <Deadman />
            </div>
        )
    }
}

var r = (
    <BrowserRouter>
        <Layout />
    </BrowserRouter>
);

render(r, document.getElementById('app'));
