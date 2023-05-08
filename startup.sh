#!/bin/sh

# Ititialization

mainmenu () {
  echo "Press 1 to update your system"
  echo "Press 2 to navigate to Pycharm file and open it"
  read -p "Input Selection:" mainmenuinput
  if [ "$mainmenuinput" = "1" ]; then
            sudo apt-get clean && sudo apt-get autoclean && sudo apt-get update && sudo apt-get upgrade && apt-get dist-upgrade && sudo apt-get autoclean && sudo apt-get autoremove --purge
        elif [ "$mainmenuinput" = "2" ]; then
	    read -p "Type in screen scale:" scaleinput
		
	    export DISPLAY=:0
	    export QT_SCALE_FACTOR=$scaleinput
	    export GDK_SCALE=$scaleinput
            sh /opt/pycharm-*/bin/pycharm.sh
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

