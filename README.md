# Holy Score bridge (WanGP plugin)

Connects your WanGP to the **Holy Score** studio (https://score.holyfstudios.com), where you write a film's music and foley to its picture.

## Install

In WanGP: **Plugin Manager → install from URL** → paste this repository's URL, enable it, and restart WanGP.

A **Holy Score** tab appears in WanGP with the address (and a public tunnel address) and the pairing key: paste both into the studio's connect dialog.

## What it gives the studio (port 7867)

- **YuE2 recordings** of your sketch (they join WanGP's normal queue, scores saved as ABC/MIDI)
- **PrismAudio foley** made from the video, shot by shot or for any stretch
- Clean speed changes (Rubber Band), joining video clips into one film, and exporting the video with its finished audio
- Project storage, humming transcription and the producer's requests

Every request needs the pairing key. Your files stay in WanGP's `outputs/holy-music` folder.
