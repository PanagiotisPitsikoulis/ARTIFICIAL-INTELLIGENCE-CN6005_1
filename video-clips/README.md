# Video Clips - Twitter & Reddit Clip Generator

Create polished Twitter/X and Reddit video clips from your screen recordings using [Remotion](https://remotion.dev).

## Features

- **Twitter/X Templates**
  - Square format (1080x1080) - perfect for Twitter feed
  - Widescreen format (1920x1080) - for Twitter video player
  - X branding with customizable accent colors
  
- **Reddit Templates**
  - Widescreen format (1920x1080) - standard Reddit video
  - Vertical format (1080x1920) - optimized for Reddit mobile
  - Reddit-style header with subreddit branding
  - Decorative upvote/comment/share bar

- **Animations**
  - Smooth intro animations with spring physics
  - Fade-out transitions
  - Title reveal animations

## Installation

```bash
cd video-clips
npm install
```

## Quick Start

### 1. Start the Remotion Studio (Preview)

```bash
npm start
```

This opens Remotion Studio at http://localhost:3000 where you can:
- Preview all compositions
- Adjust props in real-time
- Test different video sources

### 2. Add Your Screen Recording

Place your screen recording in the `public/recordings/` folder:

```
video-clips/
  public/
    recordings/
      my-screen-recording.mp4   <-- Add your file here
```

### 3. Render Your Clip

#### Using npm scripts:

```bash
# Twitter square (1:1)
npm run render:twitter-square -- --props='{"videoSrc":"recordings/my-video.mp4","title":"Check this out!"}'

# Twitter widescreen (16:9)
npm run render:twitter-wide -- --props='{"videoSrc":"recordings/my-video.mp4","title":"New feature demo"}'

# Reddit (16:9)
npm run render:reddit -- --props='{"videoSrc":"recordings/my-video.mp4","title":"My post title","subreddit":"programming"}'

# Reddit vertical (9:16)
npm run render:reddit-vertical -- --props='{"videoSrc":"recordings/my-video.mp4","title":"Mobile clip"}'
```

#### Using the render script:

```bash
npx ts-node scripts/render.ts TwitterClipSquare my-video.mp4 --title="Check this out!"
npx ts-node scripts/render.ts RedditClip demo.mp4 --subreddit="webdev" --title="New feature"
```

## Available Compositions

| Composition | Dimensions | Aspect Ratio | Best For |
|-------------|------------|--------------|----------|
| `TwitterClipSquare` | 1080x1080 | 1:1 | Twitter feed, Instagram |
| `TwitterClipWide` | 1920x1080 | 16:9 | Twitter video player |
| `RedditClip` | 1920x1080 | 16:9 | Reddit desktop |
| `RedditClipVertical` | 1080x1920 | 9:16 | Reddit mobile, TikTok, Reels |

## Props Reference

### Twitter Clips

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `videoSrc` | string | `""` | Path to video file (in public folder) or URL |
| `title` | string | `""` | Title text overlay |
| `showTitle` | boolean | `true` | Whether to show the title |
| `backgroundColor` | string | `"#000000"` | Background color |
| `accentColor` | string | `"#1DA1F2"` | Accent color (border, X logo) |
| `padding` | number | `40` | Padding around video |

### Reddit Clips

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `videoSrc` | string | `""` | Path to video file (in public folder) or URL |
| `title` | string | `""` | Post title |
| `subreddit` | string | `""` | Subreddit name (without r/) |
| `showTitle` | boolean | `true` | Whether to show title |
| `showSubreddit` | boolean | `true` | Whether to show subreddit |
| `backgroundColor` | string | `"#1A1A1B"` | Background color |
| `accentColor` | string | `"#FF4500"` | Reddit orange accent |
| `padding` | number | `40` | Padding around video |

## Customization

### Changing Video Duration

By default, compositions are 300 frames (10 seconds at 30fps). To render a different duration:

```bash
npx remotion render TwitterClipSquare out/clip.mp4 --frames=450  # 15 seconds
```

Or set duration based on input video:
```bash
npx remotion render TwitterClipSquare out/clip.mp4 --props='{"videoSrc":"recordings/my-video.mp4"}' --codec=h264
```

### Custom Colors

Pass custom colors via props:

```bash
npx remotion render TwitterClipSquare out/clip.mp4 --props='{
  "videoSrc": "recordings/demo.mp4",
  "backgroundColor": "#1a1a2e",
  "accentColor": "#e94560"
}'
```

## Output

Rendered videos are saved to the `out/` folder:

```
video-clips/
  out/
    twitter-square.mp4
    reddit.mp4
    ...
```

## Tips

1. **Video Format**: MP4 with H.264 codec works best
2. **Screen Recording Tools**: OBS, QuickTime, Loom, or built-in OS tools
3. **Optimal Length**: Keep clips under 60 seconds for social media
4. **Title Length**: Keep titles concise (under 60 characters)

## Development

```bash
# Start development studio
npm start

# Build production bundle
npm run build

# Upgrade Remotion
npm run upgrade
```

## Requirements

- Node.js 18+
- npm or yarn
- FFmpeg (for video processing)
