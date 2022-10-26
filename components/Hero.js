const Site = require("../site.json");
import { getBasePath } from "../lib/paths";
import Video from "./Video";
import Image from "./Image";

export default function Hero({
  image,
  video,
  caption,
  color,
  textColor,
  overlay,
  style,
  children,
}) {
  const bg = color ? `bg-${color}` : `bg-${Site.theme}-${Site.shade}`;
  const alt = caption || "Hero image";
  const text = textColor ? `text-${textColor}` : "text-white";
  const align = overlay ? "text-center" : "text-left";

  return (
    <div
      className={`comp_hero relative flex w-screen justify-center self-stretch  ${bg}`}
      style={style}
    >
      <div className="flex flex-col lg:flex-row max-w-screen-xl lg:w-full items-center justify-start px-10 lg:px-20 xl:px-10">
        
        <div
          className={`flex flex-1 prose-lg ${text} mr-0 lg:mr-4 pt-0 pb-10 lg:pt-40 lg:pb-40 z-20 ${
            overlay ? "justify-center" : ""
          }`}
        >
          <div className={`text-left lg:${align} mt-20 lg:mt-0 max-w-xl`}>
            {children}
          </div>
        </div>

        {!overlay && image && (
          <div className="flex flex-1 ml-0 lg:ml-4 mb-20 lg:mb-0 w-full">
            {video && <div className="w-full"><Video aspectRatio="wide" url={video} autoPlay muted loop controls={false} alt={alt} poster={image || false} /></div>}
            {!video && image && <Image url={image} alt={alt} />}
          </div>
        )}

      </div>

      {overlay && (
        <div
          className={`absolute top-0 left-0 right-0 bottom-0 opacity-10 pointer-events-none bg-cover bg-center z-10`}
          style={{ backgroundImage: `url(${image})` }}
        ></div>
      )}
    </div>
  );
}
