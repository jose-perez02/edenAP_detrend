# edenAP

This repository stores codes to perform aperture photometry on datasets from 
the EDEN survey. 

It has been adapted directly from lcogtAP, created by Néstor Espinoza (nsespino@uc.cl)

DEPENDENCIES
------------

This code makes use of the following python libraries:

    + Numpy
    + Scipy
    + Pyfits
    + Beautiful Soup (https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
    + Astropy (http://www.astropy.org/)
    + Photutils (http://photutils.readthedocs.io/)
    + pyephem (https://pypi.python.org/pypi/pyephem/)
    + Astroquery (https://astroquery.readthedocs.io/en/latest/)
    + funpack (https://heasarc.gsfc.nasa.gov/fitsio/fpack/)

All of them are open source and can be easily installed in any machine. 

Furthermore, it optionally makes use of the astrometry package (if your images 
dont have an astrometric solution or you want to compute it yourself with the pipeline), 
which can be downloaded from here: 

                http://astrometry.net/use.html. 

Installing astronometry.net on a mac can be done by following these steps:

1. Install homebrew by following instructions at: https://brew.sh/

2. Tap `brewsci` and install `astrometry-net`:

```brew tap brewsci/homebrew-science```
```brew install brewsci/homebrew-science/astrometry-net```

3. Grab the 4100 and 4200 index files from http://broiler.astrometry.net/~dstn

4.  Make sure the `astrometry.net` package knows where to find the index files.

INSTALLATION & SETUP
--------------------

The code does not need heavy installing of anything once the dependencies are installed. However, 
some setup is needed for each system and/or photometric run, and these are defined in the `setup.dat` 
file that you can fill with your information.

Under the `FOLDER OPTIONS` of the setup file, you have to fill the folders on which the `funpack` and 
the `astrometry` executables are. For `funpack` it is usually of the form `/yourpath/cfitsio/`, while 
for `astrometry` they are usually of the form `/yourpath/astrometry/bin/`.

Under the `USER OPTIONS`, you might define some user-defined properties:

    SENDEMAIL                 If set to True, an email will be sent from an user-defined e-mail address 
                              to an user-defined e-mail address with information regarding the reduced 
                              data (lightcurves and images).

    EMAILSENDER               E-mail of the e-mail sender. Currently supports only gmail accounts. 
                              Note the gmail account has to be habilitated for this via the less 
                              secure apps: https://www.google.com/settings/security/lesssecureapps

    EMAILSENDER_PASSWORD      Password of the e-mail sender defined above.

    EMAILRECEIVER             If SENDEMAIL is set to True, this is a comma-separated list of e-mails that 
                              will receive the e-mail with the information regarding the reduced data.

Under the `PHOTOMETRY OPTIONS` you can tweak what the pipeline will do:

    ASTROMETRY                If set to True, the pipeline will automatically perform the photometry.
                              For current LCOGT images this is not needed, so you might want to set it 
                              to False.

    GFASTROMETRY              If set to True, the astrometry will be performed on a copy of the original 
                              image where a gaussian filter will be performed. This greatly 
                              improves the astrometric solution on highly defocused images.

USAGE
------------

Once all of the above is installed, using the code is easy. However, it relies 
on having internet connection:

 1. First, edit the userdata.dat file and fill in a project name and a folder 
    where the data is for that project. It is assumed inside the project 
    folder there is another folder named `raw`, which contains the data 
    for different dates. For example, if the project folder for the 
    project `VATT` is `/data/VATT/`, it is assumed 
    the folder `/data/VATT/raw` exists, and contains 
    the data for different dates in separate folders (e.g., 
    `/data/VATT/raw/20180128` ).

2. From terminal, go to the pipeline folder (`cd pipeline`) and run:

       python automatic_photometry_eden.py -telescop NAMEOFTHETELESCOPE -ndays N

   Where `NAMEOFTHETELESCOPE` corresponds to the telescope in the userdata.dat file 
   that you filled in in step 1. This will run the pipeline and save the products 
   under a `red` folder, inside the telescope's folder (e.g., if the telescope was 
   `VATT`, products will be saved in `/data/VATT/red`). 
   The pipeline will generate photometry for all the datasets in the folder for which no 
   photometry is yet available that have dates (which is measured from the folder name, 
   i.e., if the dataset is in `/data/VATT/red/20180128` it assumes 
   the dataset is from 2018/01/28) whithin `N` days from today (measured from your 
   computer's date; if the `-ndays` input is not given, it is assumed `N` is `7`, i.e., 
   check data only one week appart from today).

If you are interested in what the pipeline does, read the next section. If you don't care, 
move to the "Products" section.

WHAT DOES THE PIPELINE DO?
---------------------------

In the background, what the pipeline does is to use the `pipeline/get_photometry_eden.py` 
code to get the photometry for all the stars in the field for all the images. It first 
identifies the objects in the image by doing a query to 2MASS (previous to running an 
astrometric solution with astrometry, if the images don't have it). This photometry is 
stored in a `photometry.pkl` file in the `red` folder.

Once this is done, the code calls the `post_processing/transit_photometry.py` code to generate 
differential photometry for the target star. In order to identify the target star for which 
differential photometry should be performed, the code uses the object name in the headers of 
the images and queries the Kepler webpage, Simbad, or the `manual_object_coords.dat` file in 
this folder (in that order). If the target name is not identified in any of these cases, then 
no differential photometry is performed and you have to perform it on your own. Aperture photometry 
is performed for 5 apertures: opt, 5, 15, 20 and 30 pixels. The `opt` aperture is an optimal aperture 
calculated on the rms of the lightcurve; the `get_photometry_eden.py` code does apertures from 5 
pixels to 50 pixels, and the `post_processing/transit_photometry.py` code searches for the aperture 
with the smallest RMS.

It is important to note that the differential photometry performed by the code is based on first 
combining the set of comparison stars light curves most similar to the target's, and then generating 
a "super comparison" star with those, which is finally divided into the target star light curve. This is done 
in order to not "touch" the target star, so the target could have systematic effects/trends arising 
from airmass, variability, etc. In general these are useful, because one can later detrend the 
resulting lightcurve with other methods. 

PRODUCTS
--------

The products of the `automatic_eden.py` code are stored in the `red` folder for each date and for 
each target. Inside, you will find:

 1. The `photometry.pkl` file, which contains all the aperture photometry for all the stars in the field 
    for all the apertures (5 to 50 pixels).

 2. The `post_processing` folder.

Each of the `sinistro_ap` folders contain two `.dat` files: the one with the `norm_flux.dat` sufix contains the 
times (in BJD), relative flux and error on the relative flux of the (corrected by the comparison stars) target 
star, while the other `.dat` file contains the same in differential magnitudes. 

The folder `LC` is very important, as it contains the differential magnitude of the target and the comparison 
stars in the `.epdlc` format, which is useful if you want to detrend your data or play with other ways of combining 
the extracted fluxes of the target and its comparison stars. This format contains lots of information, including: 
file name, times in BJD, relative magnitude of the target, error on the magnitude, x and y centroids, background flux, 
error on the background flux, full-width half-maximum, hour angle, and zenith angle.

The `raw_light_curves` folder contains the original extracted light curves for the target and comparison stars.

The `post_processing_outputs` folder contains the photometry for all the apertures in `.dat` files, similar to the 
`.dat` files inside the `sinistro_ap` folders. 

Finally, all the other folders contain images in png format of the target and the comparison stars.


POST-PROCESSING USAGE
----------------------

In the eventuality in which the post-processing fails, you can do it yourself inside the `post_processing` folder. 
A common usage is:

           python transit_photometry.py -telescope NAMEOFTHETELESCOPE -datafolder 20180128 -target_name TARGETNAME -ra "14:00:00" -dec " -60:00:00" --plt_images

This will save the photometry in the `red/20180128` folder inside the telescope folder `NAMEOFTHETELESCOPE` for the target 
`TARGETNAME`.

To Do
-----

1. Ensure bias subtraction, flat-fielding, bad pixel correction, and cosmic ray corrections are applied in pre-processing calibration pipeline

2. Photometry procedure occasionally give NaN's values in the reference stars light curves. Find out why!

3. In order to evaluate aperture/reference star ranking, it might be useful to include radial plots, FWHM fits plots, etc.

4. Come up with procedure to deal with nearby background stars during photometry procedure and aperture selection.
