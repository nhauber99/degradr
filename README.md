# degradr
Python library for realistically degrading images.
Work in progress, I will add more documentation when having something to show for.
<br><br>

**The applied steps are as follows (assuming the image is already in the camera color space):**
<ol>
<li>Convolve by random blur kernel (a combination of defocus blur, gaussian blur, PSFs generated from Zernike polynomials to model the lens aberrations, chromatic aberration)</li>
<li>Color filter array (in practice applied directly before the demosaicing for simplicity, but this doesn't affect the output)</li>
<li>Poison noise</li>
<li>Gain</li>
<li>Read Noise</li>
<li>Quantization</li>
<li>Camera white balance</li>
<li>Demosaicing</li>
<li>Color space transformation (from white balance corrected camera color space to sRGB)</li>
</ol>
<br><br>

**Examples:**
| Input | ![Image](Examples/in.png) |
|:-:|-|
| Blur | ![Image](Examples/blur.png) | 
| Blur <br> Noise | ![Image](Examples/noise_blur.png) |
| Blur <br> Noise <br> AHD Demosaicing | ![Image](Examples/noise_blur_ahd.png) |
| Blur <br> Noise <br> AHD Demosaicing <br> JPG Compression | ![Image](Examples/noise_blur_ahd_jpg.png) |
