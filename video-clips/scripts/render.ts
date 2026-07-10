#!/usr/bin/env npx ts-node

import { execSync } from "child_process";
import * as path from "path";
import * as fs from "fs";

interface RenderOptions {
  composition: "TwitterClipSquare" | "TwitterClipWide" | "RedditClip" | "RedditClipVertical";
  videoSrc: string;
  title?: string;
  subreddit?: string;
  output?: string;
  duration?: number;
}

function render(options: RenderOptions) {
  const { composition, videoSrc, title = "", subreddit = "", output, duration } = options;

  const outputPath = output || `out/${composition}-${Date.now()}.mp4`;

  const props: Record<string, unknown> = {
    videoSrc: videoSrc.startsWith("/") ? videoSrc : `recordings/${videoSrc}`,
    title,
  };

  if (composition.startsWith("Reddit")) {
    props.subreddit = subreddit;
    props.showSubreddit = !!subreddit;
  }

  props.showTitle = !!title;

  let cmd = `npx remotion render ${composition} ${outputPath} --props='${JSON.stringify(props)}'`;

  if (duration) {
    const frames = duration * 30;
    cmd += ` --frames=${frames}`;
  }

  console.log(`Rendering ${composition}...`);
  console.log(`Video: ${videoSrc}`);
  console.log(`Output: ${outputPath}`);
  console.log("");

  try {
    execSync(cmd, { stdio: "inherit", cwd: path.join(__dirname, "..") });
    console.log(`\nDone! Output saved to: ${outputPath}`);
  } catch (error) {
    console.error("Render failed:", error);
    process.exit(1);
  }
}

const args = process.argv.slice(2);

if (args.length === 0 || args.includes("--help")) {
  console.log(`
Video Clips Renderer

Usage:
  npx ts-node scripts/render.ts <composition> <video> [options]

Compositions:
  TwitterClipSquare   - 1080x1080 square format for Twitter/X
  TwitterClipWide     - 1920x1080 widescreen for Twitter/X
  RedditClip          - 1920x1080 widescreen for Reddit
  RedditClipVertical  - 1080x1920 vertical for Reddit mobile

Options:
  --title="Your Title"      Add a title overlay
  --subreddit="programming" Add subreddit name (Reddit only)
  --output="out/video.mp4"  Custom output path
  --duration=10             Duration in seconds (auto-detects from video if not set)

Examples:
  npx ts-node scripts/render.ts TwitterClipSquare my-recording.mp4 --title="Check this out!"
  npx ts-node scripts/render.ts RedditClip demo.mp4 --subreddit="programming" --title="New feature"
`);
  process.exit(0);
}

const composition = args[0] as RenderOptions["composition"];
const videoSrc = args[1];

const parseArg = (name: string): string | undefined => {
  const arg = args.find((a) => a.startsWith(`--${name}=`));
  return arg ? arg.split("=").slice(1).join("=") : undefined;
};

render({
  composition,
  videoSrc,
  title: parseArg("title"),
  subreddit: parseArg("subreddit"),
  output: parseArg("output"),
  duration: parseArg("duration") ? parseInt(parseArg("duration")!, 10) : undefined,
});
