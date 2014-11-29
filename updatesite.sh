#!/bin/bash
git checkout master && (
  ./cleanfiles.sh
  (
    cd doc
    make html
    cd _build/html
    touch .nojekyll
  )
  git checkout gh-pages && (
    git rm $(git ls-tree $(git log --color=never | head -n1 | cut -f2 -d' ') -r --name-only)
    cp -rv doc/_build/html/* doc/_build/html/.* .
    for f in $(find doc/_build/html | cut -c17-); do
      echo Adding $f
      git add $f
    done
    git commit --no-verify -a --amend -m 'Automatic documentation update'
    git push origin gh-pages:gh-pages -f
  )
) && git checkout master
