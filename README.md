<html>
## 📂 Qualitive comparison

|       | Train         | Test               |
| ----- | ------------- | ------------------ |
| Input | `input/*.png` | `test-input/*.png` |
| GT    | `gt/*.png`    | `test-gt/*.png`    |

**Abstract:** Existing image enhancement methods are typically designed to address either the over- or under-exposure problem in the input image. When the illumination of the input image contains both over- and under-exposure problems, these existing methods may not work well. We observe from the image statistics that the local color distributions (LCDs) of an image suffering from both problems tend to vary across different regions of the image, depending on the local illuminations. Based on this observation, we propose in this paper to exploit these LCDs as a prior for locating and enhancing the two types of regions (i.e., over-/under-exposed regions). First, we leverage the LCDs to represent these regions, and propose a novel local color distribution embedded (LCDE) module to formulate LCDs in multi-scales to model the correlations across different regions. Second, we propose a dual-illumination learning mechanism to enhance the two types of regions. Third, we construct a new dataset to facilitate the learning process, by following the camera image signal processing (ISP) pipeline to render standard RGB images with both under-/over-exposures from raw data. Extensive experiments demonstrate that the proposed method outperforms existing state-of-the-art methods quantitatively and qualitatively.

## 📂 Quantitative comparison

|               | LPIPS   | VIFs    | SSIM    | PSNR     |
| ------------- | ------- | ------- | ------- | -------- |
| Zero-DCE      | 0.254   | 0.512   | 0.756   | 17.383   |
| EnlightenGAN  | 0.231   | 0.500   | 0.768   | 19.187   |
| IAT           | 0.294   | 0.121   | 0.754   | 20.913   |
| LCDP          | 0.160   | 0.565   |<span style="color:red">0.842</span>|<span style="color:red">23.239</span>|
| Ours          |<span style="color:red">0.157</span>|<span style="color:red">0.567</span>| 0.832   | 20.942   |

<span style="color:red">빨간 글씨</span>
## 🔥 Our Model

![Our model](https://hywang99.github.io/images/lcdpnet/arch.png)

## 📂 Dataset

The LCDP Dataset is here: [[Google drive]](https://drive.google.com/drive/folders/10Reaq-N0DiZiFpSrZ8j5g3g0EJes4JiS?usp=sharing). Please unzip `lcdp_dataset.7z`. The training and test images are:

|       | Train         | Test               |
| ----- | ------------- | ------------------ |
| Input | `input/*.png` | `test-input/*.png` |
| GT    | `gt/*.png`    | `test-gt/*.png`    |

</html>
