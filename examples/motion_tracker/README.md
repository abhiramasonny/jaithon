# motion_tracker

Sparse optical flow over a scene that is generated, not filmed. jaicv picks
the corners worth following, then carries them from frame to frame and draws
where they have been. Nothing is downloaded and there is no camera — the whole
thing runs from a clean checkout.

```bash
jaithon run examples/motion_tracker/track.jai
```

About four tenths of a second for 24 frames of 640×480. It writes
`motion_trails.png`: the last frame with a trail behind every point that
survived, fading from red where it started to green where it is now.

Typical run: **104 of 120 corners still tracked after 24 frames**, while the
backdrop has turned about 10° and panned some 40 pixels.

## Why the backdrop is drawn the way it is

The scene has to be worth tracking. A flat or repeating background gives the
tracker nothing to lock onto and every point drifts off; pure noise is worse,
because noise has corners everywhere and none of them are the same corner a
frame later once the frame has turned. So the backdrop is a grainy base with a
hundred and twenty rectangles and discs over it — edges and junctions, which
survive a rotation and can be found again.

Two discs cross the scene on their own paths, against the camera's motion.
They carry no trails: the corners are chosen once, on the first frame, and
after that the program only ever asks where *those* points went. Choosing
corners again every frame would be a different program — one that could never
say that this point is that point.

## What the two numbers mean

The run prints the time following the points separately from the time drawing
the scene for them to be followed through, because only the first is the
library doing the work this example is about. The following is the larger of
the two, and it is dominated by building the image pyramid rather than by how
many points are on it — tracking four times as many costs barely more.

Points are dropped on two conditions, not one. The tracker reports whether it
lost a point off the edge of the frame, and separately how far apart the two
windows still are once it has finished; a point whose windows never really
agreed is dropped too, which is what keeps a trail from wandering off across a
flat patch of background.
