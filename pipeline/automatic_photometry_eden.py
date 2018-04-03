# -*- coding: utf-8 -*-
import numpy as np
import subprocess
import urllib
import astropy.io.fits as pyfits
import jdcal
import shutil
import glob
import os
import argparse
import PhotUtils

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')

import requests
from bs4 import BeautifulSoup

import ast
from datetime import datetime
 
import smtplib
import mimetypes
from dateutil import parser
from email.mime.multipart import MIMEMultipart
from email import encoders
from email.message import Message
from email.mime.audio import MIMEAudio
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.text import MIMEText

########################################################################
# Classes and functions

def read_setupfile():
    fin = open('../setup.dat','r')
    fpack_folder = ''
    astrometry_folder = ''
    SEND_EMAIL = False
    emailsender = ''
    emailsender_pwd = ''
    emailreceiver = ['']
    ASTROMETRY = False
    GF_ASTROMETRY = False
    done = False
    while True:
        line = fin.readline()
        if 'FOLDER OPTIONS' in line:
            while True:
                line = fin.readline()
                if 'funpack' in line:
                    fpack_folder = line.split('=')[-1].split('\n')[0].strip()
                if 'astrometry' in line:
                    astrometry_folder = line.split('=')[-1].split('\n')[0].strip()
                if 'USER OPTIONS' in line:
                    break
        if 'USER OPTIONS' in line:
            while True:
                line = fin.readline()
                line_splitted = line.split('=')
                if len(line_splitted) == 2:
                    opt,res = line_splitted
                    opt = opt.strip()
                    res = res.split('\n')[0].strip()
                    if 'SENDEMAIL' == opt:
                        if res.lower() == 'true':
                            SEND_EMAIL = True
                    if 'EMAILSENDER' == opt:
                            emailsender = res
                    if 'EMAILSENDER_PASSWORD' == opt:
                            emailsender_pwd = res
                    if 'EMAILRECEIVER' == opt:
                            emailreceiver = res.split(',')
                if 'PHOTOMETRY OPTIONS' in line:
                    break
        if 'PHOTOMETRY OPTIONS' in line:
            while True:
                line = fin.readline()
                line_splitted = line.split('=')
                if len(line_splitted) == 2:
                    opt,res = line_splitted
                    opt = opt.strip()
                    res = res.split('\n')[0].strip()
                    if opt == 'ASTROMETRY':
                        if res.lower() == 'true':
                            ASTROMETRY = True
                    if opt == 'GFASTROMETRY':
                        if res.lower() == 'true':
                            GF_ASTROMETRY = True
                if line == '':
                    done = True
                    break
        if done:
            break
    return fpack_folder,astrometry_folder,SEND_EMAIL,emailsender,emailsender_pwd,\
           emailreceiver,ASTROMETRY,GF_ASTROMETRY

class Bimail:
    def __init__(self,subject,recipients):
        fpack_folder,astrometry_folder,SEND_EMAIL,emailsender,emailsender_pwd,\
        emailreceiver,ASTROMETRY,GF_ASTROMETRY = read_setupfile()
        self.subject = subject
        self.recipients = recipients
        self.htmlbody = ''
        self.sender = emailsender 
        self.senderpass = emailsender_pwd
        self.attachments = []
 
    def send(self):
        msg = MIMEMultipart('alternative')
        msg['From']=self.sender
        msg['Subject']=self.subject
        msg['To'] = ", ".join(self.recipients) # to must be array of the form ['mailsender135@gmail.com']
        msg.preamble = "preamble goes here"
        #check if there are attachments if yes, add them
        if self.attachments:
            self.attach(msg)
        #add html body after attachments
        msg.attach(MIMEText(self.htmlbody, 'html'))
        #send
        s = smtplib.SMTP('smtp.gmail.com:587')
        s.starttls()
        s.login(self.sender,self.senderpass)
        s.sendmail(self.sender, self.recipients, msg.as_string())
        #test
        print( msg )
        s.quit()
    
    def htmladd(self, html):
        self.htmlbody = self.htmlbody+'<p></p>'+html
 
    def attach(self,msg):
        print( self.attachments )
        for f in self.attachments:
        
            ctype, encoding = mimetypes.guess_type(f)
            if ctype is None or encoding is not None:
                ctype = "application/octet-stream"
                
            maintype, subtype = ctype.split("/", 1)
 
            if maintype == "text":
                fp = open(f)
                # Note: we should handle calculating the charset
                attachment = MIMEText(fp.read(), _subtype=subtype)
                fp.close()
            elif maintype == "image":
                fp = open(f, "rb")
                attachment = MIMEImage(fp.read(), _subtype=subtype)
                fp.close()
            elif maintype == "audio":
                fp = open(f, "rb")
                attachment = MIMEAudio(fp.read(), _subtype=subtype)
                fp.close()
            else:
                fp = open(f, "rb")
                attachment = MIMEBase(maintype, subtype)
                attachment.set_payload(fp.read())
                fp.close()
                encoders.encode_base64(attachment)
            attachment.add_header("Content-Disposition", "attachment", filename=f.split('/')[-1])
            attachment.add_header('Content-ID', '<{0}>'.format(f.split('/')[-1]))
            msg.attach(attachment)
    
    def addattach(self, files):
        self.attachments = self.attachments + files

def datesok(year,month):
    if year == 2016:
        if month >= 10:
            return True
    elif year > 2016:
        return True
    return False

def login(URL, username=None, password=None):
    """
    authenticate to access HTTPS web service
    """
    s = requests.session()

    # Retrieve the CSRF token first
    s.get(URL, verify=False) # sets cookie
    csrftoken = s.cookies['csrftoken']

    login_data = dict(username=username, password=password,
        csrfmiddlewaretoken=csrftoken)
    s.post(URL, data=login_data, headers=dict(Referer=URL))
    return s

def get_login_data():
    return -1

def get_hs_coords(s,target):
    return -1

def spaced(input,space):
    fixed = False
    i = 0
    input = space+input
    while(not fixed):
        if(input[i:i+1] == '\n'):
           input = input[0:i+1]+space+input[i+1:]
           i = i + len(space)
        i = i + 1
        if(i == len(input)-1):
          fixed = True
    return input

def get_epic_coords(epicid):
    url = 'http://archive.stsci.edu/k2/epic/search.php?'
    url += 'action=Search'
    url += '&target='+epicid
    url += '&outputformat=CSV'
    lines = urllib.urlopen(url)
    data = {}
    counter = 0
    for line in lines:
        if counter == 0:
            names = line.split(',')
            names[-1] = names[-1].split('\n')[0]
            counter += 1
        elif counter == 1:
            dtypes = line.split(',')
            dtypes[-1] = dtypes[-1].split('\n')[0]
            counter += 1
        else:
            values = line.split(',')
            values[-1] = values[-1].split('\n')[0]
            for j in range(len(values)):
                if dtypes[j] == 'integer':
                    if values[j] != '':
                        data[names[j]] = int(values[j])
                    else:
                        data[names[j]] = -1
                elif dtypes[j] == 'float':
                    if values[j] != '':
                        data[names[j]] = float(values[j])
                    else:
                        data[names[j]] = -1
                else:
                    data[names[j]] = values[j]
    return data['RA'],data['Dec']

import decimal
def NumToStr(number,roundto=None):
    if roundto is not None:
        number = round(decimal.Decimal(str(number)),roundto)
    abs_number = np.abs(number)
    if abs_number < 10:
        str_number = '0'+str(abs_number)
    else:
        str_number = str(abs_number)
    if number < 0:
        return '-'+str_number
    else:
        return str_number

########################################################################
# Main procedure

# Get user input:
parserIO = argparse.ArgumentParser()
parserIO.add_argument('-telescope',default=None)
parserIO.add_argument('-ndays',default=7)
args = parserIO.parse_args()

# Get the telescope name (see the userdata.dat file):
telescope = args.telescope
ndays = int(args.ndays)

# Check for which telescope the user whishes to download data from:
ftelescopes = open('../userdata.dat','r')
while True:
    line = ftelescopes.readline()
    if line != '':
        if line[0] != '#':
            cp,cf = line.split()
            cp = cp.split()[0]
            cf = cf.split()[0]
            if telescope.lower() == cp.lower():
                break
    else:
        print( '\t > Telescope '+telescope+' is not on the list of saved telescopes. ' )
        print( '\t   Please associate it on the userdata.dat file.' )

data_folder = cf

fpack_folder,astrometry_folder,SEND_EMAIL,emailsender,emailsender_pwd,\
emailreceiver,ASTROMETRY,GF_ASTROMETRY = read_setupfile()

emails_to_send = emailreceiver #['nestor.espinozap@gmail.com','daniel.bayliss01@gmail.com','andres.jordan@gmail.com']

folders_raw = sorted(glob.glob(data_folder+'raw/*'))
dates_raw = len(folders_raw)*[[]]
for i in range(len(folders_raw)):
    dates_raw[i] = folders_raw[i].split('/')[-1]
if not os.path.exists(data_folder+'red/'):
    os.mkdir(data_folder+'red/')
folders_red = sorted(glob.glob(data_folder+'red/*'))
dates_red = len(folders_red)*[[]]
for i in range(len(folders_red)):
    dates_red[i] = folders_red[i].split('/')[-1]

# Run the get_photometry_lcogt code for all the raw folders in case new data from past nights was 
# reduced by LCOGT today. If no new data, nothing will happen (i.e., the code does nothing):
today_jd = sum(jdcal.gcal2jd(str(datetime.today().year), str(datetime.today().month), str(datetime.today().day)))
for i in range(len(dates_raw)):
    first_HS_login = True
    year = int(dates_raw[i][:4])
    month = int(dates_raw[i][4:6])
    day = int(dates_raw[i][6:8])
    s = str(year)+'.'+str(month)+'.'+str(day)
    dt = parser.parse(s)
    data_jd = sum(jdcal.gcal2jd(dt.year, dt.month, dt.day))
    # Default code only checks data one week appart (maximum LCOGT takes to process data is  
    # ~couple of days, but one week is the limit just to be sure):
    if data_jd > today_jd-ndays:
        # Get already reduced targets (if any):
        bf = sorted(glob.glob(data_folder+'red/'+dates_raw[i]+'/*'))
        before_target_folders = []
        for tar_dir in bf:
            if os.path.exists(tar_dir+'/sinistro'):
                before_target_folders.append(tar_dir)
        # Reduce the data (if already reduced, nothing will happen):
        print( '>> Reducing data for '+dates_raw[i]+' night. Reducing...' )
        optional_options = ''
        if ASTROMETRY:
            optional_options = ' --get_astrometry'
        if GF_ASTROMETRY:
            optional_options = optional_options+' --gf_opt_astrometry'
            
        os.system('python get_photometry_eden.py -telescope '+telescope+' -datafolder '+dates_raw[i]+optional_options)
        
#         continue # Post-processing algorithm below needs some work
        
########################################################################
        
        # Now, assuming it is done, run the post-processing. First, switch to the post-processing folder:
        cwd = os.getcwd()
        os.chdir('../post_processing')
        out_folder = data_folder+'red/'+dates_raw[i]+'/'
        target_folders = sorted(glob.glob(out_folder+'*'))

        # First, go through every observed object for the given night:
        for target_folder in target_folders:
            # Post-process the target only if it has already not been done:
            if target_folder in before_target_folders:
                continue
            target = target_folder.split('/')[-1]
            print( 'Post-processing target '+target+' in folder '+target_folder )
            # Try to get coordinates from Simbad or manual_coordinates.dat
            splitted_name = target.split('-')
            band = splitted_name[-1]
            target_name = '-'.join(splitted_name[:-1])
            RA, DEC = PhotUtils.get_general_coords(target_name,dates_raw[i])
            if RA == 'NoneFound':
                targetok = False
            else:
                targetok = True
            # If we can't determine RA and DEC, we can't run the post-processing algorithm
            if not targetok:
                 if SEND_EMAIL:
                    mymail = Bimail('LCOGT DR (telescope: '+telescope+'): '+target_name+' on ' +datetime.now().strftime('%Y/%m/%d'), emails_to_send)
                    mymail.htmladd('Post-processing failed for object '+target+' on '+dates_raw[i])
                    mymail.send()
                    continue
            # Assuming RA an DEC have been retrieved, run the post-processing algorithm:
            for ap in ['opt']:
                p = subprocess.Popen('echo $DISPLAY',stdout = subprocess.PIPE, \
                                     stderr = subprocess.PIPE,shell = True)
                p.wait()
                out, err = p.communicate()
                if ap == 'opt':
                    code = 'python transit_photometry.py -telescope '+telescope+' -datafolder '+\
                           dates_raw[i]+' -target_name '+target_name+' -band "'+band+\
                           '" -ra "'+RA+'" -dec "'+DEC+'" -ncomp 10 --plt_images --autosaveLC'
                else:
                    code = 'python transit_photometry.py -telescope '+telescope+' -datafolder '+\
                           dates_raw[i]+' -target_name '+target_name+' -band "'+band+\
                           '" -ra "'+RA+'" -dec "'+DEC+'" -ncomp 10 --plt_images --force_aperture -forced_aperture '+ap+' --autosaveLC'
                print( code )
                p = subprocess.Popen(code,stdout = subprocess.PIPE, \
                                     stderr = subprocess.PIPE,shell = True)
                p.wait()
                out = sorted(glob.glob(data_folder+'red/'+dates_raw[i]+'/'+target+'/*'))
#                 for ii in range(len(out)):
#                     if out[ii].split('/')[-1] == 'sinistro':
#                         out_folder = out[ii]
#                         camera = 'sinistro'
#                         break
#                     elif out[ii].split('/')[-1] == 'sbig':
#                         out_folder = out[ii]
#                         camera = 'SBIG'
#                         break
#                 shutil.move(out_folder,out_folder.rstrip('/')+'_'+ap+'/')
                if SEND_EMAIL:
                    if(p.returncode != 0 and p.returncode != None):
                        print( 'Error sending mail:' )
                        out, err = p.communicate()
                        print( spaced(err,"\t \t") )
                    print( 'Sending e-mail...' )
                    mymail = Bimail('EDEN DR (telescope: '+telescope+'): '+target_name+' on ' +dates_raw[i]+' Aperture: '+ap, emails_to_send)
                    mymail.htmladd('Data reduction was a SUCCESS! Attached is the lightcurve data.')
                    out_folder = out_folder+'_'+ap
                    real_camera = 'sinistro' # from now on, all LCOGT data comes from sinistro cameras
                    imgs = sorted(glob.glob(out_folder+'/target/*'))
  #                     f = data_folder+'raw/'+dates_raw[i]+'/'+(imgs[0].split('/')[-1]).split('.')[0]+'.fits'
  #                     d,h = pyfits.getdata(f, header=True)
                    mymail.htmladd('Camera: '+camera)
  #                     mymail.htmladd('Observing site: '+h['SITE'])
                    mymail.htmladd('Band: '+band)
                    mymail.htmladd('Dome: '+dome)
                    if len(imgs)>2:
                        mymail.addattach([imgs[0]])
                        mymail.addattach([imgs[1]])
                        mymail.addattach([imgs[2]])
                    elif len(imgs)==2:
                        mymail.addattach([imgs[0]])
                        mymail.addattach([imgs[1]])
                    else:
                        mymail.addattach([imgs[0]])
                    mymail.addattach([out_folder+'/'+target_name+'.dat'])
                    mymail.addattach([out_folder+'/'+target_name+'.pdf'])
                    mymail.addattach([out_folder+'/LC/'+target_name+'.epdlc'])
                    mymail.send()
#                 shutil.move(out_folder[:-3]+'_opt',data_folder+'red/'+dates_raw[i]+'/'+target+'/sinistro')

        # Get back to photometric pipeline directory:
        os.chdir(cwd) 
