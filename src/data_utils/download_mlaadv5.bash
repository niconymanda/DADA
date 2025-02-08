# Downloads MLAADv5 in current dir.
# Ideally run in /data/<username>/desired/path
echo "Commencing download of MLAAD V5 data"

# Download files with proper handling for redirection and filenames
wget --content-disposition -O mlaad_v5.zip.md5 "https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP/download?path=%2Fv5&files=mlaad_v5.zip.md5"
wget --content-disposition -O mlaad_v5.zip "https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP/download?path=%2Fv5&files=mlaad_v5.zip"
wget --content-disposition -O mlaad_v5.z01 "https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP/download?path=%2Fv5&files=mlaad_v5.z01"
wget --content-disposition -O mlaad_v5.z02 "https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP/download?path=%2Fv5&files=mlaad_v5.z02"
wget --content-disposition -O mlaad_v5.z03 "https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP/download?path=%2Fv5&files=mlaad_v5.z03"
wget --content-disposition -O mlaad_v5.z04 "https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP/download?path=%2Fv5&files=mlaad_v5.z04"
wget --content-disposition -O mlaad_v5.z05 "https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP/download?path=%2Fv5&files=mlaad_v5.z05"
wget --content-disposition -O mlaad_v5.z06 "https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP/download?path=%2Fv5&files=mlaad_v5.z06"
wget --content-disposition -O mlaad_v5.z07 "https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP/download?path=%2Fv5&files=mlaad_v5.z07"
wget --content-disposition -O mlaad_v5.z08 "https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP/download?path=%2Fv5&files=mlaad_v5.z08"
wget --content-disposition -O mlaad_v5.z09 "https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP/download?path=%2Fv5&files=mlaad_v5.z09"
wget --content-disposition -O mlaad_v5.z10 "https://owncloud.fraunhofer.de/index.php/s/tL2Y1FKrWiX4ZtP/download?path=%2Fv5&files=mlaad_v5.z10"

echo "Checking integrity of downloaded files"

# Ensure the MD5 file references the correct filenames
md5sum -c mlaad_v5.zip.md5

