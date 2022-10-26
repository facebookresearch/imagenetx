const Site = require("../site.json");
import Link from "next/link";

export default function Button({ url, children, color, small, style }) {
  const c = color || `${Site.theme}-${Site.shade + 100}`;
  const size = small ? "px-4 py-2 text-sm" : "px-6 py-4 text-md";

  return (
    <span className="comp_button inline-block" style={style}>
      <Link href={url} passHref>
        <a
          className={`flex rounded-md no-underline items-center justify-center font-semibold text-white hover:text-${c} hover:shadow-md bg-${c} hover:bg-white ${size}`}
        >
          {children}
        </a>
      </Link>
    </span>
  );
}
