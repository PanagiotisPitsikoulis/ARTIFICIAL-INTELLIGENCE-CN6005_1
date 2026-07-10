import {
  AbsoluteFill,
  OffthreadVideo,
  useCurrentFrame,
  useVideoConfig,
  interpolate,
  spring,
  staticFile,
  Sequence,
} from "remotion";
import { z } from "zod";

export const redditClipSchema = z.object({
  videoSrc: z.string(),
  title: z.string(),
  subreddit: z.string(),
  showTitle: z.boolean(),
  showSubreddit: z.boolean(),
  backgroundColor: z.string(),
  accentColor: z.string(),
  padding: z.number(),
});

type RedditClipProps = z.infer<typeof redditClipSchema>;

export const RedditClip: React.FC<RedditClipProps> = ({
  videoSrc,
  title,
  subreddit,
  showTitle,
  showSubreddit,
  backgroundColor,
  accentColor,
  padding,
}) => {
  const frame = useCurrentFrame();
  const { fps, width, height, durationInFrames } = useVideoConfig();

  const isVertical = height > width;

  const introProgress = spring({
    frame,
    fps,
    config: {
      damping: 200,
    },
  });

  const fadeOut = interpolate(
    frame,
    [durationInFrames - 15, durationInFrames],
    [1, 0],
    { extrapolateLeft: "clamp", extrapolateRight: "clamp" }
  );

  const videoScale = interpolate(introProgress, [0, 1], [0.9, 1]);
  const videoOpacity = interpolate(introProgress, [0, 1], [0, 1]);

  const resolvedVideoSrc = videoSrc.startsWith("http")
    ? videoSrc
    : staticFile(videoSrc);

  const headerHeight = isVertical ? 120 : 80;

  return (
    <AbsoluteFill
      style={{
        backgroundColor,
        opacity: fadeOut,
      }}
    >
      {/* Reddit Header */}
      {(showTitle || showSubreddit) && (
        <Sequence from={5}>
          <div
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              right: 0,
              height: headerHeight,
              backgroundColor: "#272729",
              display: "flex",
              alignItems: "center",
              padding: `0 ${padding}px`,
              gap: 16,
              transform: `translateY(${interpolate(introProgress, [0, 1], [-headerHeight, 0])}px)`,
              borderBottom: `3px solid ${accentColor}`,
            }}
          >
            {/* Reddit Logo */}
            <svg
              width={isVertical ? 48 : 40}
              height={isVertical ? 48 : 40}
              viewBox="0 0 24 24"
              fill={accentColor}
            >
              <circle cx="12" cy="12" r="10" fill={accentColor} />
              <ellipse cx="8.5" cy="11" rx="1.5" ry="2" fill="white" />
              <ellipse cx="15.5" cy="11" rx="1.5" ry="2" fill="white" />
              <circle cx="8.5" cy="10.5" r="0.5" fill={accentColor} />
              <circle cx="15.5" cy="10.5" r="0.5" fill={accentColor} />
              <path
                d="M8 14.5 Q12 17 16 14.5"
                stroke="white"
                strokeWidth="1.5"
                fill="none"
                strokeLinecap="round"
              />
              <ellipse cx="18" cy="5" rx="2" ry="2" fill={accentColor} />
              <line x1="14" y1="4" x2="17" y2="5" stroke={accentColor} strokeWidth="1.5" />
            </svg>

            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {showSubreddit && subreddit && (
                <span
                  style={{
                    color: accentColor,
                    fontSize: isVertical ? 24 : 18,
                    fontWeight: 700,
                    fontFamily: "system-ui, -apple-system, sans-serif",
                  }}
                >
                  r/{subreddit}
                </span>
              )}
              {showTitle && title && (
                <span
                  style={{
                    color: "#D7DADC",
                    fontSize: isVertical ? 20 : 16,
                    fontWeight: 500,
                    fontFamily: "system-ui, -apple-system, sans-serif",
                    maxWidth: width - padding * 2 - 100,
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                  }}
                >
                  {title}
                </span>
              )}
            </div>
          </div>
        </Sequence>
      )}

      {/* Video Container */}
      <AbsoluteFill
        style={{
          top: (showTitle || showSubreddit) ? headerHeight : 0,
          padding: isVertical ? `${padding / 2}px` : `${padding}px`,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <div
          style={{
            width: "100%",
            height: `calc(100% - ${(showTitle || showSubreddit) ? headerHeight : 0}px)`,
            borderRadius: 12,
            overflow: "hidden",
            transform: `scale(${videoScale})`,
            opacity: videoOpacity,
            boxShadow: `0 10px 40px rgba(0, 0, 0, 0.6)`,
          }}
        >
          {videoSrc && (
            <OffthreadVideo
              src={resolvedVideoSrc}
              style={{
                width: "100%",
                height: "100%",
                objectFit: "contain",
                backgroundColor: "#000",
              }}
            />
          )}
          {!videoSrc && (
            <div
              style={{
                width: "100%",
                height: "100%",
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
                backgroundColor: "#0e0e0f",
                color: "#666",
                fontSize: isVertical ? 28 : 24,
                fontFamily: "system-ui, sans-serif",
              }}
            >
              Drop your screen recording here
            </div>
          )}
        </div>
      </AbsoluteFill>

      {/* Upvote/Interaction Bar (decorative) */}
      <Sequence from={20}>
        <div
          style={{
            position: "absolute",
            bottom: padding,
            left: padding,
            display: "flex",
            alignItems: "center",
            gap: 20,
            opacity: interpolate(
              spring({ frame: frame - 20, fps, config: { damping: 200 } }),
              [0, 1],
              [0, 0.7]
            ),
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill={accentColor}>
              <path d="M12 4l-8 8h5v8h6v-8h5z" />
            </svg>
            <span style={{ color: "#818384", fontSize: 14, fontFamily: "system-ui" }}>
              Vote
            </span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="#818384">
              <path d="M21 6h-2V3a1 1 0 0 0-1-1H6a1 1 0 0 0-1 1v3H3a1 1 0 0 0-1 1v13a1 1 0 0 0 1 1h18a1 1 0 0 0 1-1V7a1 1 0 0 0-1-1zM7 4h10v2H7V4zm13 15H4V8h16v11z" />
            </svg>
            <span style={{ color: "#818384", fontSize: 14, fontFamily: "system-ui" }}>
              Comment
            </span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="#818384">
              <path d="M18 16.08c-.76 0-1.44.3-1.96.77L8.91 12.7c.05-.23.09-.46.09-.7s-.04-.47-.09-.7l7.05-4.11c.54.5 1.25.81 2.04.81 1.66 0 3-1.34 3-3s-1.34-3-3-3-3 1.34-3 3c0 .24.04.47.09.7L8.04 9.81C7.5 9.31 6.79 9 6 9c-1.66 0-3 1.34-3 3s1.34 3 3 3c.79 0 1.5-.31 2.04-.81l7.12 4.16c-.05.21-.08.43-.08.65 0 1.61 1.31 2.92 2.92 2.92s2.92-1.31 2.92-2.92-1.31-2.92-2.92-2.92z" />
            </svg>
            <span style={{ color: "#818384", fontSize: 14, fontFamily: "system-ui" }}>
              Share
            </span>
          </div>
        </div>
      </Sequence>
    </AbsoluteFill>
  );
};
