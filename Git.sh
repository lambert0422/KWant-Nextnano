#!/bin/sh

# Ititialization

mainmenu () {
  echo "Press 1 to add files"
  echo "Press 2 to commit"
  echo "Press 3 to Push"
  echo "Press 4 to Pull"
  read -p "Input Selection:" mainmenuinput
  if [ "$mainmenuinput" = "1" ]; then
            echo "Press 1 to add all"
	    echo "Press 2 to add specific file"
	    read -p "Input Selection:" Addinput
	    if [ "$Addinput" = "1" ]; then
		    git add -A
	    elif [ "$Addinput" = "2" ]; then
		    read -p "Type in file name:" Filename
		    git add "$Filename"
    	    fi
        elif [ "$mainmenuinput" = "2" ]; then
            read -p "Type in Commit message: " Message
	    git commit -m "$Message"
	elif [ "$mainmenuinput" = "3" ]; then
		git push
	elif [ "$mainmenuinput" = "4" ]; then
		read -p "Git url:" URL
		read -p "Branch name:" Branch
		git pull "$URL" "$Branch"
        else
            echo "You have entered an invallid selection!"
            echo "Please try again!"
            echo ""
            echo "Press any key to continue..."
            read -n 1
            clear
            mainmenu
        fi
}

# This builds the main menu and routs the user to the function selected.

mainmenu


# This executes the main menu function.
# Let the fun begin!!!! WOOT WOOT!!!!
