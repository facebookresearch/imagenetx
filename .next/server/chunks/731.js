"use strict";
exports.id = 731;
exports.ids = [731];
exports.modules = {

/***/ 8731:
/***/ ((module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.a(module, async (__webpack_handle_async_dependencies__, __webpack_async_result__) => { try {
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "Z": () => (/* binding */ Nav)
/* harmony export */ });
/* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(997);
/* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var _headlessui_react__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(1185);
/* harmony import */ var next_link__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(1664);
/* harmony import */ var next_image__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(5675);
/* harmony import */ var _lib_paths__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(4437);
/* harmony import */ var react_icons_ri__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(8098);
/* harmony import */ var react_icons_ri__WEBPACK_IMPORTED_MODULE_4___default = /*#__PURE__*/__webpack_require__.n(react_icons_ri__WEBPACK_IMPORTED_MODULE_4__);
var __webpack_async_dependencies__ = __webpack_handle_async_dependencies__([_headlessui_react__WEBPACK_IMPORTED_MODULE_1__]);
_headlessui_react__WEBPACK_IMPORTED_MODULE_1__ = (__webpack_async_dependencies__.then ? (await __webpack_async_dependencies__)() : __webpack_async_dependencies__)[0];







const Site = __webpack_require__(5272);
function Nav({ items , selected  }) {
    const color = Site.theme;
    const logoWithLink = (logo)=>{
        return Site.showHomeLink ? /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx("div", {
            className: "flex-shrink-0 flex items-center cursor-pointer",
            children: logo
        }) : /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(next_link__WEBPACK_IMPORTED_MODULE_2__["default"], {
            href: `/home`,
            passHref: true,
            children: /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx("div", {
                className: "flex-shrink-0 flex items-center cursor-pointer",
                children: logo
            })
        });
    };
    return /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(_headlessui_react__WEBPACK_IMPORTED_MODULE_1__.Disclosure, {
        as: "nav",
        className: "bg-white md:shadow",
        children: ({ open  })=>/*#__PURE__*/ (0,react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.Fragment, {
                children: [
                    /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx("div", {
                        className: "max-w-screen-xl mx-auto px-4 sm:px-6 lg:px-8",
                        children: /*#__PURE__*/ (0,react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)("div", {
                            className: "flex justify-between h-16",
                            children: [
                                /*#__PURE__*/ (0,react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)("div", {
                                    className: "flex",
                                    children: [
                                        logoWithLink(/*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(next_image__WEBPACK_IMPORTED_MODULE_3__["default"], {
                                            src: Site.logo,
                                            alt: "logo",
                                            height: 32,
                                            width: 32,
                                            loader: _lib_paths__WEBPACK_IMPORTED_MODULE_5__/* .assetLoader */ .H
                                        })),
                                        /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx("div", {
                                            className: "hidden sm:ml-6 sm:flex sm:space-x-8",
                                            children: items && items.map((item)=>Site.showHomeLink || item.id !== "home" ? /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(next_link__WEBPACK_IMPORTED_MODULE_2__["default"], {
                                                    href: `/${item.id}`,
                                                    passHref: true,
                                                    children: /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx("a", {
                                                        className: `${selected === item.id ? `text-${color}-700` : ""} hover:text-${color}-500 text-gray-900 inline-flex items-center px-1 pt-1 text-sm font-semibold`,
                                                        children: item.title
                                                    })
                                                }, item.id) : null
                                            )
                                        })
                                    ]
                                }),
                                /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx("div", {
                                    className: "hidden sm:ml-6 sm:flex sm:space-x-8",
                                    children: Site.rightNav && Site.rightNav.map((item)=>/*#__PURE__*/ (0,react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)("a", {
                                            href: item.url,
                                            className: `text-gray-900 hover:text-${color}-500 ml-6 inline-flex items-center px-1 pt-1 text-sm font-semibold`,
                                            target: "_blank",
                                            children: [
                                                item.title,
                                                "\xa0",
                                                /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(react_icons_ri__WEBPACK_IMPORTED_MODULE_4__.RiArrowRightUpLine, {})
                                            ]
                                        }, item.title)
                                    )
                                }),
                                /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx("div", {
                                    className: "-mr-2 flex items-center sm:hidden",
                                    children: /*#__PURE__*/ (0,react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(_headlessui_react__WEBPACK_IMPORTED_MODULE_1__.Disclosure.Button, {
                                        className: `inline-flex items-center justify-center p-2 rounded-md text-gray-900`,
                                        children: [
                                            /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx("span", {
                                                className: "sr-only",
                                                children: "Open main menu"
                                            }),
                                            open ? /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(react_icons_ri__WEBPACK_IMPORTED_MODULE_4__.RiCloseCircleFill, {
                                                className: "block h-6 w-6",
                                                "aria-hidden": "true"
                                            }) : /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(react_icons_ri__WEBPACK_IMPORTED_MODULE_4__.RiMenuFill, {
                                                className: "block h-6 w-6",
                                                "aria-hidden": "true"
                                            })
                                        ]
                                    })
                                })
                            ]
                        })
                    }),
                    /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(_headlessui_react__WEBPACK_IMPORTED_MODULE_1__.Disclosure.Panel, {
                        className: "sm:hidden",
                        children: ({ close  })=>/*#__PURE__*/ (0,react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)("div", {
                                className: "pt-2 pb-3 space-y-1 absolute left-0 right-0 z-50 bg-white shadow-xl",
                                children: [
                                    items && items.map((item)=>Site.showHomeLink || item.id !== "home" ? /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(next_link__WEBPACK_IMPORTED_MODULE_2__["default"], {
                                            href: `/${item.id}`,
                                            passHref: true,
                                            prefetch: true,
                                            children: /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(_headlessui_react__WEBPACK_IMPORTED_MODULE_1__.Disclosure.Button, {
                                                as: "a",
                                                onClick: async ()=>setTimeout(close, 500)
                                                ,
                                                className: `${selected === item.id ? `bg-${color}-50 text-${color}-700` : "text-gray-900"} block pl-5 pr-4 py-2 text-base font-semibold`,
                                                children: item.title
                                            })
                                        }, item.id) : null
                                    ),
                                    Site.rightNav && Site.rightNav.map((item)=>/*#__PURE__*/ (0,react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)(_headlessui_react__WEBPACK_IMPORTED_MODULE_1__.Disclosure.Button, {
                                            as: "a",
                                            href: item.url,
                                            className: "text-gray-900 block pl-5 pr-4 py-2 border-none text-base font-semibold",
                                            children: [
                                                item.title,
                                                /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx("span", {
                                                    className: "inline-block align-middle",
                                                    children: /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(react_icons_ri__WEBPACK_IMPORTED_MODULE_4__.RiArrowRightUpLine, {})
                                                })
                                            ]
                                        }, item.title)
                                    )
                                ]
                            })
                    })
                ]
            })
    });
};

__webpack_async_result__();
} catch(e) { __webpack_async_result__(e); } });

/***/ }),

/***/ 4437:
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {

/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "H": () => (/* binding */ assetLoader),
/* harmony export */   "b": () => (/* binding */ getBasePath)
/* harmony export */ });
/*
 * Since `next export` currently have issues with image loader and paths,
 * here we define a custom function to resolve path.
 * Caveat: Images are not automatically optimized like, as they would be if deployed with Next.js
 *
 * The base path for development is always "/"
 * For production, you can define the basePath in .env.production
 *
 * You can also access the current base path via process.env.basePath
 */ function assetLoader({ src  }) {
    return `${"/site"}/assets/${src}`;
}
function getBasePath(src) {
    // When basePath is "", Next.js automatically prepend "/" but not when basePath is "/xyz".
    // Here we account for this condition
    return  false ? 0 : `${"/site"}${src}`;
}


/***/ }),

/***/ 5272:
/***/ ((module) => {

module.exports = JSON.parse('{"title":"ImageNet-X","logo":"logo.png","description":"","showHomeLink":true,"thumbnail":"/assets/thumbnail.png","theme":"indigo","shade":800,"paper":"white","footerBackground":"slate-200","footerText":"slate-600","rightNav":[{"title":"Github","url":"https://github.com/facebookresearch/imagenetx/"}]}');

/***/ })

};
;