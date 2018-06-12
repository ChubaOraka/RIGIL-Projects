git pull

set /p message="What message do you want to add to your commit: "
git add .
git commit -m %message%

git push

pause