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

## What the timing depends on

The run prints the time spent following the points separately from the total,
because only the first is the library doing the work this example is about,
and the first frame of all pays for compiling the kernels.

Two things move that number, and neither is obvious:

- **How many points.** Below about a hundred it is flat — 30 points and 120
  points both cost about 2.8 ms on a 640×480 pair, because there are not
  enough of them to give the device anything to do and the image pyramid is
  the whole bill. Past that it scales: 480 points cost 9.0 ms and 1920 cost
  14.2 ms.
- **How far things moved.** Each point refines its own guess and stops early
  once the guess settles, so a frame pair that barely moved converges on the
  first pass. The same 120 points cost 2.9 ms on a pair shifted by two pixels
  and 15.6 ms here, where the scene has turned as well as panned.

Points are dropped on two conditions, not one. The tracker reports whether it
lost a point off the edge of the frame, and separately how far apart the two
windows still are once it has finished; a point whose windows never really
agreed is dropped too, which is what keeps a trail from wandering off across a
flat patch of background.
