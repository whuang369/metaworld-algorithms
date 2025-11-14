f=metaworld-algorithms

cd ../.. # cd just outside the repo
tar --exclude="chtc" \
    --exclude=".git" \
    --exclude="dro_examples/run_results" \
    -czvf ${f}.tar.gz $f

scp ${f}.tar.gz ncorrado@ap2001.chtc.wisc.edu:/staging/ncorrado
rm ${f}.tar.gz