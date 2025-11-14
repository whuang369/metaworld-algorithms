f=metaworld-algorithms

cd ../.. # cd just outside the repo
tar --exclude="chtc" \
    --exclude=".DS_Store" \
    --exclude=".gitignore" \
    --exclude=".venv" \
    --exclude=".python-version" \
    --exclude=".git" \
    --exclude=".idea" \
    --exclude="metaworld/.gitignore" \
    --exclude="metaworld/.github" \
    --exclude="metaworld/.git" \
    --exclude="metaworld/metaworld.egg-info" \
    --exclude="metaworld/docs" \
    -czvf ${f}.tar.gz $f

scp ${f}.tar.gz whuang369@ap2001.chtc.wisc.edu:/staging/whuang369
rm ${f}.tar.gz