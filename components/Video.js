import { getBasePath } from "../lib/paths";

export default function Video({
  url,
  youtubeId,
  aspectRatio = "wide",
  controls = true,
  autoPlay = false,
  loop = true,
  muted = true,
  poster,
  alt = "",
  style,
}) {
  const aspect =
    aspectRatio === "wide"
      ? `aspect-w-16 aspect-h-9`
      : aspectRatio === "square"
      ? "aspect-w-9 aspect-h-9"
      : "aspect-w-4 aspect-h-3";

  return youtubeId ? (
    <div className={`comp_video w-full ${aspect} mt-3`} style={style}>
      <iframe
        src={`https://www.youtube.com/embed/${youtubeId}?&autoplay=${
          autoPlay ? 1 : 0
        }&controls=${controls ? 1 : 0}&mute=${muted}`}
        frameBorder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowFullScreen
        title={alt}
      />
    </div>
  ) : (
    <div className={`comp_video w-full ${aspect} mt-3`} style={style}>
      <video
        controls={controls}
        autoPlay={autoPlay}
        loop={loop}
        muted={muted}
        alt={alt}
        poster={poster}
        className="m-0"
      >
        <source src={getBasePath(url)} type="video/mp4" />
        Sorry, your browser doesn't support embedded videos.
      </video>
    </div>
  );
}
