#!/bin/sh

# Ititialization

mainmenu () {
  echo "Press 1 to add files"
  echo "Press 2 to Pull"
  read -p "Input Selection:" mainmenuinput
  if [ "$mainmenuinput" = "1" ]; then
            echo "Press 1 to add all"
	    echo "Press 2 to add specific file"
	    echo "Press 3 to add the usual files"
	    read -p "Input Selection:" Addinput
	    if [ "$Addinput" = "1" ]; then
		    git add -A
	    elif [ "$Addinput" = "2" ]; then
		    read -p "Type in file name:" Filename
		    git add "$Filename"
	    elif [ "$Addinput" = "3" ]; then
		    git add Kwant_Class_latest_ROG.py

    	    fi
	    read -p "Type in Commit message: " Message
	    git commit -m "$Message"
	    read -p "Push?[1Y/2N]: " PushYN
	    if [ "$PushYN" = "1" ]; then
		git push
	    fi

	elif [ "$mainmenuinput" = "2" ]; then
	    echo "Press 1 to pull specific url"
	    echo "Press 2 to pull usual one"
	    read -p "Input Selection:" Urlinput
	    if [ "$Urlinput" = "1" ]; then
		    read -p "Git url:" URL
		    read -p "Branch name:" Branch
		    git pull "$URL" "$Branch"
	    elif [ "$Urlinput" = "2" ]; then
		    git pull https://github.com/lambert0422/KWant-Nextnano.git ROG
	    fi
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
