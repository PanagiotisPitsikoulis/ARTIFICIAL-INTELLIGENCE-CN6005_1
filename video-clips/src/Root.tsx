import { Composition } from "remotion";
import { TwitterClip, twitterClipSchema } from "./compositions/TwitterClip";
import { RedditClip, redditClipSchema } from "./compositions/RedditClip";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {/* Twitter/X Square Format - 1:1 aspect ratio */}
      <Composition
        id="TwitterClipSquare"
        component={TwitterClip}
        durationInFrames={300}
        fps={30}
        width={1080}
        height={1080}
        schema={twitterClipSchema}
        defaultProps={{
          videoSrc: "",
          title: "",
          showTitle: true,
          backgroundColor: "#000000",
          accentColor: "#1DA1F2",
          padding: 40,
        }}
      />

      {/* Twitter/X Landscape Format - 16:9 aspect ratio */}
      <Composition
        id="TwitterClipWide"
        component={TwitterClip}
        durationInFrames={300}
        fps={30}
        width={1920}
        height={1080}
        schema={twitterClipSchema}
        defaultProps={{
          videoSrc: "",
          title: "",
          showTitle: true,
          backgroundColor: "#000000",
          accentColor: "#1DA1F2",
          padding: 40,
        }}
      />

      {/* Reddit Video Format - 16:9 */}
      <Composition
        id="RedditClip"
        component={RedditClip}
        durationInFrames={300}
        fps={30}
        width={1920}
        height={1080}
        schema={redditClipSchema}
        defaultProps={{
          videoSrc: "",
          title: "",
          subreddit: "",
          showTitle: true,
          showSubreddit: true,
          backgroundColor: "#1A1A1B",
          accentColor: "#FF4500",
          padding: 40,
        }}
      />

      {/* Reddit Mobile/Vertical Format - 9:16 */}
      <Composition
        id="RedditClipVertical"
        component={RedditClip}
        durationInFrames={300}
        fps={30}
        width={1080}
        height={1920}
        schema={redditClipSchema}
        defaultProps={{
          videoSrc: "",
          title: "",
          subreddit: "",
          showTitle: true,
          showSubreddit: true,
          backgroundColor: "#1A1A1B",
          accentColor: "#FF4500",
          padding: 40,
        }}
      />
    </>
  );
};
