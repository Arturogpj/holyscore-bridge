# Holy Score bridge (WanGP plugin)

Connects your WanGP to the **Holy Score** studio (https://score.holyfstudios.com), where you write a film's music and foley to its picture.

## Install

1. In WanGP, open the **Plugins** tab.
2. Under **Install New Plugin**, paste `https://github.com/Arturogpj/holyscore-bridge` and press **Download and Install Plugin**.
3. Turn the plugin on and restart WanGP.
4. Open the new **Holy Score** tab and copy the **pairing key**.
5. Open https://score.holyfstudios.com, click **WanGP · not connected** (top right), paste the key and press **Connect**.

Same PC as WanGP: that's all. Another computer: use the internet address shown in the Holy Score tab (under "WanGP is on another computer" in the studio's Setup).

To update later: the same Plugins tab updates it (then restart WanGP).

## What it gives the studio (port 7867)

- **YuE2 recordings** of your sketch (they join WanGP's normal queue, scores saved as ABC/MIDI)
- **PrismAudio foley** made from the video, shot by shot or for any stretch
- Clean speed changes (Rubber Band), joining video clips into one film, and exporting the video with its finished audio
- Project storage, humming transcription and the producer's requests

Every request needs the pairing key. Your files stay in WanGP's `outputs/holy-music` folder.
