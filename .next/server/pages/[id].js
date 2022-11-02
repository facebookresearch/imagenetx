"use strict";
(() => {
var exports = {};
exports.id = 112;
exports.ids = [112];
exports.modules = {

/***/ 8634:
/***/ ((__unused_webpack_module, __webpack_exports__, __webpack_require__) => {


// EXPORTS
__webpack_require__.d(__webpack_exports__, {
  "Z": () => (/* binding */ _export)
});

// EXTERNAL MODULE: external "react/jsx-runtime"
var jsx_runtime_ = __webpack_require__(997);
// EXTERNAL MODULE: ./lib/paths.js
var paths = __webpack_require__(4437);
;// CONCATENATED MODULE: ./components/Video.js


function Video({ url , youtubeId , aspectRatio ="wide" , controls =true , autoPlay =false , loop =true , muted =true , poster , alt ="" , style ,  }) {
    const aspect = aspectRatio === "wide" ? `aspect-w-16 aspect-h-9` : aspectRatio === "square" ? "aspect-w-9 aspect-h-9" : "aspect-w-4 aspect-h-3";
    return youtubeId ? /*#__PURE__*/ jsx_runtime_.jsx("div", {
        className: `comp_video w-full ${aspect} mt-3`,
        style: style,
        children: /*#__PURE__*/ jsx_runtime_.jsx("iframe", {
            src: `https://www.youtube.com/embed/${youtubeId}?&autoplay=${autoPlay ? 1 : 0}&controls=${controls ? 1 : 0}&mute=${muted}`,
            frameBorder: "0",
            allow: "accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture",
            allowFullScreen: true,
            title: alt
        })
    }) : /*#__PURE__*/ jsx_runtime_.jsx("div", {
        className: `comp_video w-full ${aspect} mt-3`,
        style: style,
        children: /*#__PURE__*/ (0,jsx_runtime_.jsxs)("video", {
            controls: controls,
            autoPlay: autoPlay,
            loop: loop,
            muted: muted,
            alt: alt,
            poster: poster,
            className: "m-0",
            children: [
                /*#__PURE__*/ jsx_runtime_.jsx("source", {
                    src: (0,paths/* getBasePath */.b)(url),
                    type: "video/mp4"
                }),
                "Sorry, your browser doesn't support embedded videos."
            ]
        })
    });
};

;// CONCATENATED MODULE: ./components/Image.js


function Image({ url , caption , contain , style , showCaption  }) {
    const resize = contain ? "object-contain" : "object-cover w-full h-full";
    return /*#__PURE__*/ (0,jsx_runtime_.jsxs)("div", {
        className: "relative",
        children: [
            /*#__PURE__*/ jsx_runtime_.jsx("img", {
                src: (0,paths/* getBasePath */.b)(url),
                alt: caption,
                className: `comp_image ${resize} m-0`,
                style: style
            }),
            caption && showCaption && /*#__PURE__*/ jsx_runtime_.jsx("div", {
                className: "absolute bottom-0 left-0 right-0 bg-black bg-opacity-50 text-white text-xs p-3 md:p-4",
                children: caption
            })
        ]
    });
};

;// CONCATENATED MODULE: ./components/Hero.js

const Site = __webpack_require__(5272);



function Hero({ image , video , caption , color , textColor , overlay , style , children ,  }) {
    const bg = color ? `bg-${color}` : `bg-${Site.theme}-${Site.shade}`;
    const alt = caption || "Hero image";
    const text = textColor ? `text-${textColor}` : "text-white";
    const align = overlay ? "text-center" : "text-left";
    return /*#__PURE__*/ (0,jsx_runtime_.jsxs)("div", {
        className: `comp_hero relative flex w-screen justify-center self-stretch  ${bg}`,
        style: style,
        children: [
            /*#__PURE__*/ (0,jsx_runtime_.jsxs)("div", {
                className: "flex flex-col lg:flex-row max-w-screen-xl lg:w-full items-center justify-start px-10 lg:px-20 xl:px-10",
                children: [
                    /*#__PURE__*/ jsx_runtime_.jsx("div", {
                        className: `flex flex-1 prose-lg ${text} mr-0 lg:mr-4 pt-0 pb-10 lg:pt-40 lg:pb-40 z-20 ${overlay ? "justify-center" : ""}`,
                        children: /*#__PURE__*/ jsx_runtime_.jsx("div", {
                            className: `text-left lg:${align} mt-20 lg:mt-0 max-w-xl`,
                            children: children
                        })
                    }),
                    !overlay && image && /*#__PURE__*/ (0,jsx_runtime_.jsxs)("div", {
                        className: "flex flex-1 ml-0 lg:ml-4 mb-20 lg:mb-0 w-full",
                        children: [
                            video && /*#__PURE__*/ jsx_runtime_.jsx("div", {
                                className: "w-full",
                                children: /*#__PURE__*/ jsx_runtime_.jsx(Video, {
                                    aspectRatio: "wide",
                                    url: video,
                                    autoPlay: true,
                                    muted: true,
                                    loop: true,
                                    controls: false,
                                    alt: alt,
                                    poster: image || false
                                })
                            }),
                            !video && image && /*#__PURE__*/ jsx_runtime_.jsx(Image, {
                                url: image,
                                alt: alt
                            })
                        ]
                    })
                ]
            }),
            overlay && /*#__PURE__*/ jsx_runtime_.jsx("div", {
                className: `absolute top-0 left-0 right-0 bottom-0 opacity-10 pointer-events-none bg-cover bg-center z-10`,
                style: {
                    backgroundImage: `url(${image})`
                }
            })
        ]
    });
};

// EXTERNAL MODULE: ./node_modules/next/link.js
var next_link = __webpack_require__(1664);
// EXTERNAL MODULE: external "react-icons/ri"
var ri_ = __webpack_require__(8098);
;// CONCATENATED MODULE: ./components/Button.js

const Button_Site = __webpack_require__(5272);


function Button({ url , children , color , small , openNew , style  }) {
    const c = color || `${Button_Site.theme}-${Button_Site.shade + 100}`;
    const size = small ? "px-4 py-2 text-sm" : "px-6 py-4 text-md";
    const classes = `flex rounded-md no-underline items-center ${openNew ? 'gap-2' : ''} justify-center font-semibold text-white hover:text-${c} hover:shadow-md bg-${c} hover:bg-white ${size}`;
    return /*#__PURE__*/ (0,jsx_runtime_.jsxs)("span", {
        className: "comp_button inline-block",
        style: style,
        children: [
            !openNew && /*#__PURE__*/ jsx_runtime_.jsx(next_link["default"], {
                href: url,
                passHref: true,
                children: /*#__PURE__*/ jsx_runtime_.jsx("a", {
                    className: classes,
                    children: children
                })
            }),
            openNew && /*#__PURE__*/ (0,jsx_runtime_.jsxs)("a", {
                href: url,
                target: "_blank",
                rel: "noopener",
                className: classes,
                children: [
                    children,
                    " ",
                    /*#__PURE__*/ jsx_runtime_.jsx(ri_.RiArrowRightUpLine, {})
                ]
            })
        ]
    });
};

;// CONCATENATED MODULE: ./components/Content.js

const Content_Site = __webpack_require__(5272);

function Content({ color , whiteText , noteLeft , noteRight , imageLeft , imageRight , spaceTop , spaceBottom , small , style , children ,  }) {
    const bgColor = color ? color === "theme" ? `bg-${Content_Site.theme}-${Content_Site.shade}` : `bg-${color}` : `bg-${Content_Site.paper}`;
    const textColor = whiteText ? "text-white" : "text-gray-700";
    const padTop = spaceTop ? "pt-20" : "pt-4";
    const padBottom = spaceBottom ? "pb-20" : "pb-4";
    const textSize = small ? "prose-sm" : "prose-lg";
    return /*#__PURE__*/ jsx_runtime_.jsx("div", {
        className: `comp_content flex w-screen justify-center self-stretch ${bgColor} ${textColor} ${whiteText ? "color-flip" : ""}`,
        style: style,
        children: /*#__PURE__*/ (0,jsx_runtime_.jsxs)("div", {
            className: `flex flex-1 flex-row box-border max-w-screen-xl items-center justify-start px-5 md:px-20 xl:px-10 ${padTop} ${padBottom}`,
            children: [
                /*#__PURE__*/ (0,jsx_runtime_.jsxs)("div", {
                    className: "hidden md:flex flex-1 flex-col mr-4 pt-2 items-start justify-start h-full text-sm",
                    children: [
                        imageLeft && /*#__PURE__*/ jsx_runtime_.jsx("div", {
                            className: "aspect-w-4 aspect-h-4",
                            children: /*#__PURE__*/ jsx_runtime_.jsx("img", {
                                src: (0,paths/* getBasePath */.b)(imageLeft),
                                alt: noteLeft || "",
                                className: "object-contain"
                            })
                        }),
                        /*#__PURE__*/ jsx_runtime_.jsx("div", {
                            className: `opacity-70 ${textColor}`,
                            children: noteLeft || ""
                        })
                    ]
                }),
                /*#__PURE__*/ jsx_runtime_.jsx("div", {
                    className: `flex-1 flex-grow-4 self-start max-w-none ${textSize} mx-4 ${textColor}`,
                    children: children
                }),
                /*#__PURE__*/ (0,jsx_runtime_.jsxs)("div", {
                    className: "hidden md:flex flex-1 flex-col ml-4 pt-2 items-start justify-start h-full text-sm",
                    children: [
                        imageRight && /*#__PURE__*/ jsx_runtime_.jsx("div", {
                            className: "aspect-auto",
                            children: /*#__PURE__*/ jsx_runtime_.jsx("img", {
                                src: (0,paths/* getBasePath */.b)(imageRight),
                                alt: noteRight || "",
                                className: "object-contain mb-2 mt-2"
                            })
                        }),
                        /*#__PURE__*/ jsx_runtime_.jsx("div", {
                            className: `flex opacity-70 ${textColor}`,
                            children: noteRight || ""
                        })
                    ]
                })
            ]
        })
    });
};

// EXTERNAL MODULE: external "react"
var external_react_ = __webpack_require__(6689);
var external_react_default = /*#__PURE__*/__webpack_require__.n(external_react_);
;// CONCATENATED MODULE: ./lib/util.js
function* chunkArray(arr, n) {
    for(let i = 0; i < arr.length; i += n){
        yield arr.slice(i, i + n);
    }
}
function zipArray(arrays) {
    let z = [];
    for(let i = 0, len = arrays[0].length; i < len; i++){
        let p = [];
        for(let k = 0; k < arrays.length; k++){
            p.push(arrays[k][i]);
        }
        z.push(p);
    }
    return z;
}

;// CONCATENATED MODULE: ./components/Gallery.js

const Gallery_Site = __webpack_require__(5272);


function Gallery({ color , whiteText , columns =1 , flowVertical , grid , fullWidth , noteLeft , noteRight , spaceTop , spaceBottom , style , children ,  }) {
    const bgColor = color ? `bg-${color}` : `bg-${Gallery_Site.paper}`;
    const textColor = whiteText ? "text-white" : "text-gray-700";
    const padTop = spaceTop ? "pt-20" : "pt-4";
    const padBottom = spaceBottom ? "pb-20" : "pb-4";
    const itemsCount = external_react_default().Children.count(children);
    const isSingle = itemsCount < 2;
    const cols = itemsCount / columns;
    const chunks = isSingle ? [] : flowVertical ? [
        ...chunkArray(children, cols)
    ] : zipArray([
        ...chunkArray(children, children.length / cols)
    ]);
    return /*#__PURE__*/ jsx_runtime_.jsx("div", {
        className: `comp_gallery flex w-screen justify-center self-stretch ${bgColor} ${textColor}`,
        style: style,
        children: /*#__PURE__*/ (0,jsx_runtime_.jsxs)("div", {
            className: `max-w-screen-xl flex flex-1 flex-row items-center justify-start px-5 md:px-20 xl:px-10 ${padTop} ${padBottom}`,
            children: [
                !fullWidth && /*#__PURE__*/ jsx_runtime_.jsx("div", {
                    className: "hidden md:flex flex-1 flex-col mr-4 pt-4 items-start justify-start h-full text-sm",
                    children: /*#__PURE__*/ jsx_runtime_.jsx("div", {
                        className: `opacity-70 ${textColor}`,
                        children: noteLeft || ""
                    })
                }),
                /*#__PURE__*/ (0,jsx_runtime_.jsxs)("div", {
                    className: `flex flex-1 flex-grow-4 self-start max-w-none ${fullWidth ? "" : "mx-4"}`,
                    children: [
                        !isSingle && chunks.map((chs, k)=>/*#__PURE__*/ jsx_runtime_.jsx("div", {
                                className: "flex-col flex-1",
                                children: chs.map((c, i)=>/*#__PURE__*/ jsx_runtime_.jsx("div", {
                                        className: `p-1 ${grid ? "aspect-[1/1]" : "aspect-auto"}`,
                                        children: c
                                    }, `img${i}`)
                                )
                            }, `group${k}`)
                        ),
                        isSingle && /*#__PURE__*/ jsx_runtime_.jsx("div", {
                            className: "p-1 aspect-auto",
                            children: children
                        })
                    ]
                }),
                !fullWidth && /*#__PURE__*/ jsx_runtime_.jsx("div", {
                    className: "hidden md:flex flex-1 flex-col ml-4 pb-4 items-start justify-end h-full text-sm",
                    children: /*#__PURE__*/ jsx_runtime_.jsx("div", {
                        className: `opacity-70 ${textColor}`,
                        children: noteRight || ""
                    })
                })
            ]
        })
    });
};

// EXTERNAL MODULE: external "better-react-mathjax"
var external_better_react_mathjax_ = __webpack_require__(1445);
;// CONCATENATED MODULE: ./components/Equation.js


function Equation({ children , style  }) {
    return /*#__PURE__*/ jsx_runtime_.jsx("div", {
        className: "comp_equation flex w-full items-center justify-center",
        style: style,
        children: /*#__PURE__*/ jsx_runtime_.jsx(external_better_react_mathjax_.MathJax, {
            children: children
        })
    });
};

;// CONCATENATED MODULE: ./components/Footer.js

const Footer_Site = __webpack_require__(5272);


function Footer({ style , columns , children  }) {
    return /*#__PURE__*/ jsx_runtime_.jsx("div", {
        className: `comp_footer flex w-screen justify-center self-stretch bg-${Footer_Site.footerBackground} text-${Footer_Site.textColor}`,
        style: style,
        children: /*#__PURE__*/ (0,jsx_runtime_.jsxs)("div", {
            className: `flex flex-1 flex-row box-border max-w-screen-xl items-start justify-start px-5 md:px-20 xl:px-10 py-20 `,
            children: [
                /*#__PURE__*/ jsx_runtime_.jsx("div", {
                    className: "hidden md:flex flex-1 flex-col mr-4 pt-2"
                }),
                columns ? external_react_default().Children.map(children, (c, i)=>{
                    return /*#__PURE__*/ jsx_runtime_.jsx("div", {
                        className: "flex flex-1 box-border pr-2 item-start justify-start text-sm",
                        children: c
                    });
                }) : children,
                /*#__PURE__*/ jsx_runtime_.jsx("div", {
                    className: "hidden md:flex flex-1 flex-col mr-4 pt-2 "
                })
            ]
        })
    });
};

;// CONCATENATED MODULE: ./components/internal/ColorScheme.js

const colors = [
    "slate",
    "gray",
    "zinc",
    "neutral",
    "stone",
    "red",
    "orange",
    "amber",
    "yellow",
    "lime",
    "green",
    "emerald",
    "teal",
    "cyan",
    "sky",
    "blue",
    "indigo",
    "violet",
    "purple",
    "fuchsia",
    "pink",
    "rose", 
];
const scales = [
    "50",
    "100",
    "200",
    "300",
    "400",
    "500",
    "600",
    "700",
    "800",
    "900", 
];
function ColorScheme() {
    return /*#__PURE__*/ jsx_runtime_.jsx(jsx_runtime_.Fragment, {
        children: colors.map((c)=>/*#__PURE__*/ jsx_runtime_.jsx("div", {
                children: /*#__PURE__*/ jsx_runtime_.jsx(ColorRow, {
                    name: c
                })
            }, c)
        )
    });
}
function ColorRow({ name  }) {
    return /*#__PURE__*/ jsx_runtime_.jsx(jsx_runtime_.Fragment, {
        children: scales.map((s)=>/*#__PURE__*/ (0,jsx_runtime_.jsxs)("span", {
                className: `bg-${name}-${s} ${parseInt(s) < 400 ? "text-slate-500" : "text-white"} block p-1 m-[1px] text-[9px] md:w-[8%] w-[7vw] inline-block md:p-1`,
                children: [
                    name,
                    /*#__PURE__*/ jsx_runtime_.jsx("br", {}),
                    s
                ]
            }, name + s)
        )
    });
}
function ColorThemeBlock({ one , two , three , name  }) {
    return /*#__PURE__*/ (0,jsx_runtime_.jsxs)("div", {
        children: [
            /*#__PURE__*/ jsx_runtime_.jsx("div", {
                className: "mt-5 mb-1 text-sm text-slate-600",
                children: name
            }),
            /*#__PURE__*/ (0,jsx_runtime_.jsxs)("div", {
                className: "flex flex-row",
                children: [
                    /*#__PURE__*/ jsx_runtime_.jsx("div", {
                        className: `flex flex-1 flex-grow-3 items-center justify-center bg-${one} text-white box-border p-10`,
                        children: one
                    }),
                    /*#__PURE__*/ jsx_runtime_.jsx("div", {
                        className: `flex flex-1 items-center justify-center bg-${two} text-white py-10`,
                        children: two
                    }),
                    /*#__PURE__*/ jsx_runtime_.jsx("div", {
                        className: `flex flex-1 items-center justify-center bg-${three} text-slate-900 py-10`,
                        children: three
                    })
                ]
            })
        ]
    });
}

;// CONCATENATED MODULE: ./components/_export.js










/* harmony default export */ const _export = ({
    Hero: Hero,
    Content: Content,
    Button: Button,
    Gallery: Gallery,
    Image: Image,
    Equation: Equation,
    Video: Video,
    Footer: Footer,
    ColorScheme: ColorScheme,
    ColorThemeBlock: ColorThemeBlock,
    Link: next_link["default"]
});


/***/ }),

/***/ 1577:
/***/ ((module, __webpack_exports__, __webpack_require__) => {

__webpack_require__.a(module, async (__webpack_handle_async_dependencies__, __webpack_async_result__) => { try {
__webpack_require__.r(__webpack_exports__);
/* harmony export */ __webpack_require__.d(__webpack_exports__, {
/* harmony export */   "getStaticPaths": () => (/* binding */ getStaticPaths),
/* harmony export */   "getStaticProps": () => (/* binding */ getStaticProps),
/* harmony export */   "default": () => (/* binding */ Post)
/* harmony export */ });
/* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__ = __webpack_require__(997);
/* harmony import */ var react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0___default = /*#__PURE__*/__webpack_require__.n(react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__);
/* harmony import */ var fs__WEBPACK_IMPORTED_MODULE_1__ = __webpack_require__(7147);
/* harmony import */ var fs__WEBPACK_IMPORTED_MODULE_1___default = /*#__PURE__*/__webpack_require__.n(fs__WEBPACK_IMPORTED_MODULE_1__);
/* harmony import */ var path__WEBPACK_IMPORTED_MODULE_2__ = __webpack_require__(1017);
/* harmony import */ var path__WEBPACK_IMPORTED_MODULE_2___default = /*#__PURE__*/__webpack_require__.n(path__WEBPACK_IMPORTED_MODULE_2__);
/* harmony import */ var gray_matter__WEBPACK_IMPORTED_MODULE_3__ = __webpack_require__(8076);
/* harmony import */ var gray_matter__WEBPACK_IMPORTED_MODULE_3___default = /*#__PURE__*/__webpack_require__.n(gray_matter__WEBPACK_IMPORTED_MODULE_3__);
/* harmony import */ var next_mdx_remote_serialize__WEBPACK_IMPORTED_MODULE_4__ = __webpack_require__(4818);
/* harmony import */ var next_mdx_remote__WEBPACK_IMPORTED_MODULE_5__ = __webpack_require__(9961);
/* harmony import */ var remark_gfm__WEBPACK_IMPORTED_MODULE_6__ = __webpack_require__(6809);
/* harmony import */ var next_head__WEBPACK_IMPORTED_MODULE_7__ = __webpack_require__(968);
/* harmony import */ var next_head__WEBPACK_IMPORTED_MODULE_7___default = /*#__PURE__*/__webpack_require__.n(next_head__WEBPACK_IMPORTED_MODULE_7__);
/* harmony import */ var _components_Nav__WEBPACK_IMPORTED_MODULE_8__ = __webpack_require__(8731);
/* harmony import */ var _components_export__WEBPACK_IMPORTED_MODULE_9__ = __webpack_require__(8634);
/* harmony import */ var _lib_paths__WEBPACK_IMPORTED_MODULE_10__ = __webpack_require__(4437);
var __webpack_async_dependencies__ = __webpack_handle_async_dependencies__([next_mdx_remote_serialize__WEBPACK_IMPORTED_MODULE_4__, next_mdx_remote__WEBPACK_IMPORTED_MODULE_5__, remark_gfm__WEBPACK_IMPORTED_MODULE_6__, _components_Nav__WEBPACK_IMPORTED_MODULE_8__]);
([next_mdx_remote_serialize__WEBPACK_IMPORTED_MODULE_4__, next_mdx_remote__WEBPACK_IMPORTED_MODULE_5__, remark_gfm__WEBPACK_IMPORTED_MODULE_6__, _components_Nav__WEBPACK_IMPORTED_MODULE_8__] = __webpack_async_dependencies__.then ? (await __webpack_async_dependencies__)() : __webpack_async_dependencies__);







const rehypePrism = __webpack_require__(5780);




const Site = __webpack_require__(5272);
const postsDirectory = path__WEBPACK_IMPORTED_MODULE_2___default().join(process.cwd(), "sections");
function getAllPostIds() {
    const fileNames = fs__WEBPACK_IMPORTED_MODULE_1___default().readdirSync(postsDirectory);
    // Returns an array that looks like this:
    // [
    //   { params: { id: 'abc' } },
    //   { params: { id: 'xyz' } }, ...
    // ]
    return fileNames.map((fileName)=>{
        return {
            params: {
                id: fileName.replace(/\.mdx?$/, "")
            }
        };
    });
}
function getAllMetaData() {
    // Read all files under /sections and get their metadata
    const fileNames = fs__WEBPACK_IMPORTED_MODULE_1___default().readdirSync(postsDirectory);
    const allMetaData = fileNames.map((fileName)=>{
        // Remove ".md" from file name to get id
        const id = fileName.replace(/\.mdx?$/, "");
        // Read markdown file as string
        const fullPath = path__WEBPACK_IMPORTED_MODULE_2___default().join(postsDirectory, fileName);
        const fileContents = fs__WEBPACK_IMPORTED_MODULE_1___default().readFileSync(fullPath, "utf8");
        // Use gray-matter to parse the post metadata section
        const { data  } = gray_matter__WEBPACK_IMPORTED_MODULE_3___default()(fileContents);
        // Combine the data with the id
        return {
            id,
            ...data
        };
    });
    // sort by frontmatter 'order'
    return allMetaData.filter(({ order  })=>order >= 0
    ).sort(({ order: a  }, { order: b  })=>{
        if (a < b) {
            return -1;
        } else if (a > b) {
            return 1;
        } else {
            return 0;
        }
    });
}
async function getStaticPaths() {
    const paths = getAllPostIds();
    return {
        paths,
        fallback: false
    };
}
const getStaticProps = async ({ params: { id  }  })=>{
    // Parse current MDX file
    const markdownWithMeta = fs__WEBPACK_IMPORTED_MODULE_1___default().readFileSync(path__WEBPACK_IMPORTED_MODULE_2___default().join("sections", id + ".mdx"), "utf-8");
    const { data , content  } = gray_matter__WEBPACK_IMPORTED_MODULE_3___default()(markdownWithMeta);
    data.id = id;
    const mdxSource = await (0,next_mdx_remote_serialize__WEBPACK_IMPORTED_MODULE_4__.serialize)(content, {
        mdxOptions: {
            remarkPlugins: [
                remark_gfm__WEBPACK_IMPORTED_MODULE_6__["default"]
            ],
            rehypePlugins: [
                rehypePrism
            ]
        }
    });
    const allMetaData = getAllMetaData();
    return {
        props: {
            id,
            mdxSource,
            currentMetaData: data,
            allMetaData
        }
    };
};
function Post({ currentMetaData , allMetaData , mdxSource  }) {
    return /*#__PURE__*/ (0,react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)("div", {
        children: [
            /*#__PURE__*/ (0,react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)((next_head__WEBPACK_IMPORTED_MODULE_7___default()), {
                children: [
                    /*#__PURE__*/ (0,react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsxs)("title", {
                        children: [
                            Site.title,
                            ": ",
                            currentMetaData.title
                        ]
                    }),
                    /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx("meta", {
                        property: "og:title",
                        content: Site.title
                    }),
                    /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx("meta", {
                        property: "og:site_name",
                        content: Site.title
                    }),
                    /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx("meta", {
                        property: "og:description",
                        content: Site.description
                    }),
                    /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx("meta", {
                        property: "og:type",
                        content: "website"
                    }),
                    /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx("meta", {
                        property: "og:image",
                        content: (0,_lib_paths__WEBPACK_IMPORTED_MODULE_10__/* .getBasePath */ .b)(Site.thumbnail)
                    }),
                    /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx("meta", {
                        name: "twitter:card",
                        content: "summary_large_image"
                    })
                ]
            }),
            /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(_components_Nav__WEBPACK_IMPORTED_MODULE_8__/* ["default"] */ .Z, {
                items: allMetaData,
                selected: currentMetaData.id
            }),
            /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx("div", {
                className: "prose prose-starter",
                children: /*#__PURE__*/ react_jsx_runtime__WEBPACK_IMPORTED_MODULE_0__.jsx(next_mdx_remote__WEBPACK_IMPORTED_MODULE_5__.MDXRemote, {
                    ...mdxSource,
                    components: _components_export__WEBPACK_IMPORTED_MODULE_9__/* ["default"] */ .Z
                })
            })
        ]
    });
};

__webpack_async_result__();
} catch(e) { __webpack_async_result__(e); } });

/***/ }),

/***/ 5780:
/***/ ((module) => {

module.exports = require("@mapbox/rehype-prism");

/***/ }),

/***/ 1445:
/***/ ((module) => {

module.exports = require("better-react-mathjax");

/***/ }),

/***/ 8076:
/***/ ((module) => {

module.exports = require("gray-matter");

/***/ }),

/***/ 562:
/***/ ((module) => {

module.exports = require("next/dist/server/denormalize-page-path.js");

/***/ }),

/***/ 4957:
/***/ ((module) => {

module.exports = require("next/dist/shared/lib/head.js");

/***/ }),

/***/ 4014:
/***/ ((module) => {

module.exports = require("next/dist/shared/lib/i18n/normalize-locale-path.js");

/***/ }),

/***/ 744:
/***/ ((module) => {

module.exports = require("next/dist/shared/lib/image-config-context.js");

/***/ }),

/***/ 5843:
/***/ ((module) => {

module.exports = require("next/dist/shared/lib/image-config.js");

/***/ }),

/***/ 8524:
/***/ ((module) => {

module.exports = require("next/dist/shared/lib/is-plain-object.js");

/***/ }),

/***/ 8020:
/***/ ((module) => {

module.exports = require("next/dist/shared/lib/mitt.js");

/***/ }),

/***/ 4964:
/***/ ((module) => {

module.exports = require("next/dist/shared/lib/router-context.js");

/***/ }),

/***/ 3938:
/***/ ((module) => {

module.exports = require("next/dist/shared/lib/router/utils/format-url.js");

/***/ }),

/***/ 9565:
/***/ ((module) => {

module.exports = require("next/dist/shared/lib/router/utils/get-asset-path-from-route.js");

/***/ }),

/***/ 4365:
/***/ ((module) => {

module.exports = require("next/dist/shared/lib/router/utils/get-middleware-regex.js");

/***/ }),

/***/ 1428:
/***/ ((module) => {

module.exports = require("next/dist/shared/lib/router/utils/is-dynamic.js");

/***/ }),

/***/ 1292:
/***/ ((module) => {

module.exports = require("next/dist/shared/lib/router/utils/parse-relative-url.js");

/***/ }),

/***/ 979:
/***/ ((module) => {

module.exports = require("next/dist/shared/lib/router/utils/querystring.js");

/***/ }),

/***/ 6052:
/***/ ((module) => {

module.exports = require("next/dist/shared/lib/router/utils/resolve-rewrites.js");

/***/ }),

/***/ 4226:
/***/ ((module) => {

module.exports = require("next/dist/shared/lib/router/utils/route-matcher.js");

/***/ }),

/***/ 5052:
/***/ ((module) => {

module.exports = require("next/dist/shared/lib/router/utils/route-regex.js");

/***/ }),

/***/ 9232:
/***/ ((module) => {

module.exports = require("next/dist/shared/lib/utils.js");

/***/ }),

/***/ 968:
/***/ ((module) => {

module.exports = require("next/head");

/***/ }),

/***/ 6689:
/***/ ((module) => {

module.exports = require("react");

/***/ }),

/***/ 8098:
/***/ ((module) => {

module.exports = require("react-icons/ri");

/***/ }),

/***/ 997:
/***/ ((module) => {

module.exports = require("react/jsx-runtime");

/***/ }),

/***/ 1185:
/***/ ((module) => {

module.exports = import("@headlessui/react");;

/***/ }),

/***/ 9961:
/***/ ((module) => {

module.exports = import("next-mdx-remote");;

/***/ }),

/***/ 4818:
/***/ ((module) => {

module.exports = import("next-mdx-remote/serialize");;

/***/ }),

/***/ 6809:
/***/ ((module) => {

module.exports = import("remark-gfm");;

/***/ }),

/***/ 7147:
/***/ ((module) => {

module.exports = require("fs");

/***/ }),

/***/ 1017:
/***/ ((module) => {

module.exports = require("path");

/***/ })

};
;

// load runtime
var __webpack_require__ = require("../webpack-runtime.js");
__webpack_require__.C(exports);
var __webpack_exec__ = (moduleId) => (__webpack_require__(__webpack_require__.s = moduleId))
var __webpack_exports__ = __webpack_require__.X(0, [895,61,731], () => (__webpack_exec__(1577)));
module.exports = __webpack_exports__;

})();