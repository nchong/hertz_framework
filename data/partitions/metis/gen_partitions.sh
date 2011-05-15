INPUT=$1
FROM=$2
TO=$3
STEP=2

PMETIS=/data/nyc04/metis-4.0.3/pmetis 

if [ ! -e $INPUT ]; then
  echo "${INPUT} not found"
  exit
fi

for npart in `seq ${FROM} ${STEP} ${TO}`; do
  ${PMETIS} ${INPUT} ${npart};
  echo ${npart} > tmp; cat $INPUT.part.${npart} >> tmp; mv -f tmp $INPUT.part.${npart};
done
