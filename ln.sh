rm -rf /var/cache/apt/archives
mkdir -p "/dev/shm/partial"
ln -s "/dev/shm" /var/cache/apt/archives
