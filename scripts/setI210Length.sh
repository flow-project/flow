#!/bin/bash
# run this like so: ./setI210Length.sh "/path/to/flow/examples/exp_configs/templates/sumo/i210_with_ghost_cell_with_downstream.xml" DESIRED_LENGTH

# Argument parsing

XML_FILE=$1   # path to the xml file
LENGTH=$2     # desired length

# Some constants

X0_EDGE="2148.42"        # original starting x value for the edge
Y0_EDGE="1244.27"        # original starting y value for the edge
X0_LANE0="2155.88"       # original starting x value for the first lane
Y0_LANE0="1239.42"       # original starting y value for the first lane
X0_LANE1="2154.23"       # original starting x value for the second lane
Y0_LANE1="1242.16"       # original starting y value for the second lane
X0_LANE2="2152.58"       # original starting x value for the third lane
Y0_LANE2="1244.91"       # original starting y value for the third lane
X0_LANE3="2150.94"       # original starting x value for the fourth lane
Y0_LANE3="1247.65"       # original starting y value for the fourth lane
X0_LANE4="2149.29"       # original starting x value for the fifth lane
Y0_LANE4="1250.39"       # original starting y value for the fifth lane
X0_LANE5="2147.64"       # original starting x value for the sixth lane
Y0_LANE5="1253.13"       # original starting y value for the sixth lane
DX="0.85"                # original (x2-x1) / length
DY="0.52"                # original (y2-y1) / length

# Parsing operation.

X1_EDGE=$(bc <<< "scale=2; $X0_EDGE + $LENGTH * $DX")
Y1_EDGE=$(bc <<< "scale=2; $Y0_EDGE + $LENGTH * $DY")
X1_LANE0=$(bc <<< "scale=2; $X0_LANE0 + $LENGTH * $DX")
Y1_LANE0=$(bc <<< "scale=2; $Y0_LANE0 + $LENGTH * $DY")
X1_LANE1=$(bc <<< "scale=2; $X0_LANE1 + $LENGTH * $DX")
Y1_LANE1=$(bc <<< "scale=2; $Y0_LANE1 + $LENGTH * $DY")
X1_LANE2=$(bc <<< "scale=2; $X0_LANE2 + $LENGTH * $DX")
Y1_LANE2=$(bc <<< "scale=2; $Y0_LANE2 + $LENGTH * $DY")
X1_LANE3=$(bc <<< "scale=2; $X0_LANE3 + $LENGTH * $DX")
Y1_LANE3=$(bc <<< "scale=2; $Y0_LANE3 + $LENGTH * $DY")
X1_LANE4=$(bc <<< "scale=2; $X0_LANE4 + $LENGTH * $DX")
Y1_LANE4=$(bc <<< "scale=2; $Y0_LANE4 + $LENGTH * $DY")
X1_LANE5=$(bc <<< "scale=2; $X0_LANE5 + $LENGTH * $DX")
Y1_LANE5=$(bc <<< "scale=2; $Y0_LANE5 + $LENGTH * $DY")

OLD_EDGE="    <edge id=\"119257908#3\" from=\"1842086610\" to=\"632089468\" name=\"Ventura Freeway\" priority=\"13\" type=\"highway.motorway\" spreadType=\"center\" shape=\"2148.42,1244.27 *"
OLD_LANE0="   <lane id=\"119257908#3_0\" index=\"0\" allow=\"private emergency authority army vip passenger hov taxi bus coach delivery truck trailer motorcycle evehicle custom1 custom2\" speed=\"5\" length=*"
OLD_LANE1="        <lane id=\"119257908#3_1\" index=\"1\" allow=\"private emergency authority army vip passenger hov taxi bus coach delivery truck trailer motorcycle evehicle custom1 custom2\" speed=\"5\" length=*"
OLD_LANE2="        <lane id=\"119257908#3_2\" index=\"2\" allow=\"private emergency authority army vip passenger hov taxi bus coach delivery truck trailer motorcycle evehicle custom1 custom2\" speed=\"5\" length=*"
OLD_LANE3="        <lane id=\"119257908#3_3\" index=\"3\" allow=\"private emergency authority army vip passenger hov taxi bus coach delivery truck trailer motorcycle evehicle custom1 custom2\" speed=\"5\" length=*"
OLD_LANE4="        <lane id=\"119257908#3_4\" index=\"4\" allow=\"private emergency authority army vip passenger hov taxi bus coach delivery truck trailer motorcycle evehicle custom1 custom2\" speed=\"5\" length=*"
OLD_LANE5="        <lane id=\"119257908#3_5\" index=\"5\" allow=\"private emergency authority army vip passenger hov taxi bus coach delivery truck trailer motorcycle evehicle custom1 custom2\" speed=\"5\" length=*"

NEW_EDGE="    <edge id=\"119257908#3\" from=\"1842086610\" to=\"632089468\" name=\"Ventura Freeway\" priority=\"13\" type=\"highway.motorway\" spreadType=\"center\" shape=\"2148.42,1244.27 "$X1_EDGE","$Y1_EDGE"\">"
NEW_LANE0="        <lane id=\"119257908#3_0\" index=\"0\" allow=\"private emergency authority army vip passenger hov taxi bus coach delivery truck trailer motorcycle evehicle custom1 custom2\" speed=\"5\" length=\""$LENGTH"\" acceleration=\"1\" shape=\"2152.67,1237.20 "$X1_LANE0","$Y1_LANE0"\">"
NEW_LANE1="        <lane id=\"119257908#3_1\" index=\"1\" allow=\"private emergency authority army vip passenger hov taxi bus coach delivery truck trailer motorcycle evehicle custom1 custom2\" speed=\"5\" length=\""$LENGTH"\" shape=\"2150.97,1240.03 "$X1_LANE1","$Y1_LANE1"\">"
NEW_LANE2="        <lane id=\"119257908#3_2\" index=\"2\" allow=\"private emergency authority army vip passenger hov taxi bus coach delivery truck trailer motorcycle evehicle custom1 custom2\" speed=\"5\" length=\""$LENGTH"\" shape=\"2149.27,1242.86 "$X1_LANE2","$Y1_LANE2"\">"
NEW_LANE3="        <lane id=\"119257908#3_3\" index=\"3\" allow=\"private emergency authority army vip passenger hov taxi bus coach delivery truck trailer motorcycle evehicle custom1 custom2\" speed=\"5\" length=\""$LENGTH"\" shape=\"2147.57,1245.68 "$X1_LANE3","$Y1_LANE3"\">"
NEW_LANE4="        <lane id=\"119257908#3_4\" index=\"4\" allow=\"private emergency authority army vip passenger hov taxi bus coach delivery truck trailer motorcycle evehicle custom1 custom2\" speed=\"5\" length=\""$LENGTH"\" shape=\"2145.87,1248.51 "$X1_LANE4","$Y1_LANE4"\">"
NEW_LANE5="        <lane id=\"119257908#3_5\" index=\"5\" allow=\"private emergency authority army vip passenger hov taxi bus coach delivery truck trailer motorcycle evehicle custom1 custom2\" speed=\"5\" length=\""$LENGTH"\" shape=\"2144.17,1251.34 "$X1_LANE5","$Y1_LANE5"\">"

sed -i '' '/'"${OLD_EDGE}"'/s/.*/'"${NEW_EDGE}"'/' "${XML_FILE}"
sed -i '' '/'"${OLD_LANE0}"'/s/.*/'"${NEW_LANE0}"'/' "${XML_FILE}"
sed -i '' '/'"${OLD_LANE1}"'/s/.*/'"${NEW_LANE1}"'/' "${XML_FILE}"
sed -i '' '/'"${OLD_LANE2}"'/s/.*/'"${NEW_LANE2}"'/' "${XML_FILE}"
sed -i '' '/'"${OLD_LANE3}"'/s/.*/'"${NEW_LANE3}"'/' "${XML_FILE}"
sed -i '' '/'"${OLD_LANE4}"'/s/.*/'"${NEW_LANE4}"'/' "${XML_FILE}"
sed -i '' '/'"${OLD_LANE5}"'/s/.*/'"${NEW_LANE5}"'/' "${XML_FILE}"