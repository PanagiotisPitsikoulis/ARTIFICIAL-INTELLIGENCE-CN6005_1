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

export const twitterClipSchema = z.object({
  videoSrc: z.string(),
  title: z.string(),
  showTitle: z.boolean(),
  backgroundColor: z.string(),
  accentColor: z.string(),
  padding: z.number(),
});

type TwitterClipProps = z.infer<typeof twitterClipSchema>;

export const TwitterClip: React.FC<TwitterClipProps> = ({
  videoSrc,
  title,
  showTitle,
  backgroundColor,
  accentColor,
  padding,
}) => {
  const frame = useCurrentFrame();
  const { fps, width, height, durationInFrames } = useVideoConfig();

  const titleProgress = spring({
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

  const videoScale = interpolate(titleProgress, [0, 1], [0.9, 1]);
  const videoOpacity = interpolate(titleProgress, [0, 1], [0, 1]);

  const resolvedVideoSrc = videoSrc.startsWith("http")
    ? videoSrc
    : staticFile(videoSrc);

  return (
    <AbsoluteFill
      style={{
        backgroundColor,
        padding,
        opacity: fadeOut,
      }}
    >
      {/* Video Container */}
      <AbsoluteFill
        style={{
          padding,
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
        }}
      >
        <div
          style={{
            width: "100%",
            height: showTitle ? "85%" : "100%",
            borderRadius: 16,
            overflow: "hidden",
            transform: `scale(${videoScale})`,
            opacity: videoOpacity,
            boxShadow: `0 20px 60px rgba(0, 0, 0, 0.5)`,
            border: `3px solid ${accentColor}`,
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
                backgroundColor: "#1a1a1a",
                color: "#666",
                fontSize: 24,
                fontFamily: "system-ui, sans-serif",
              }}
            >
              Drop your screen recording here
            </div>
          )}
        </div>
      </AbsoluteFill>

      {/* Title Overlay */}
      {showTitle && title && (
        <Sequence from={10}>
          <div
            style={{
              position: "absolute",
              bottom: padding + 10,
              left: padding,
              right: padding,
              display: "flex",
              justifyContent: "center",
            }}
          >
            <div
              style={{
                backgroundColor: "rgba(0, 0, 0, 0.85)",
                padding: "16px 32px",
                borderRadius: 12,
                border: `2px solid ${accentColor}`,
                transform: `translateY(${interpolate(titleProgress, [0, 1], [50, 0])}px)`,
                opacity: titleProgress,
              }}
            >
              <span
                style={{
                  color: "#fff",
                  fontSize: width > 1200 ? 28 : 22,
                  fontWeight: 600,
                  fontFamily: "system-ui, -apple-system, sans-serif",
                  textAlign: "center",
                }}
              >
                {title}
              </span>
            </div>
          </div>
        </Sequence>
      )}

      {/* Twitter/X Branding Watermark */}
      <div
        style={{
          position: "absolute",
          top: padding + 10,
          right: padding + 10,
          opacity: 0.7,
        }}
      >
        <svg
          width="32"
          height="32"
          viewBox="0 0 24 24"
          fill={accentColor}
        >
          <path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z" />
        </svg>
      </div>
    </AbsoluteFill>
  );
};
