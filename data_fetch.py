{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "651e1b5f-e9ae-4291-a03f-a7b06667e58b",
   "metadata": {},
   "outputs": [
    {
     "ename": "ValueError",
     "evalue": "If using all scalar values, you must pass an index",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[1], line 24\u001b[0m\n\u001b[1;32m     21\u001b[0m     \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m✅ Saved odds.csv\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m     23\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;18m__name__\u001b[39m \u001b[38;5;241m==\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m__main__\u001b[39m\u001b[38;5;124m\"\u001b[39m:\n\u001b[0;32m---> 24\u001b[0m     \u001b[43mfetch_cfb_games\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     25\u001b[0m     fetch_odds()\n",
      "Cell \u001b[0;32mIn[1], line 12\u001b[0m, in \u001b[0;36mfetch_cfb_games\u001b[0;34m()\u001b[0m\n\u001b[1;32m     10\u001b[0m r \u001b[38;5;241m=\u001b[39m requests\u001b[38;5;241m.\u001b[39mget(CFB_API)\n\u001b[1;32m     11\u001b[0m games \u001b[38;5;241m=\u001b[39m r\u001b[38;5;241m.\u001b[39mjson()\n\u001b[0;32m---> 12\u001b[0m df \u001b[38;5;241m=\u001b[39m \u001b[43mpd\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mDataFrame\u001b[49m\u001b[43m(\u001b[49m\u001b[43mgames\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     13\u001b[0m df\u001b[38;5;241m.\u001b[39mto_csv(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdata/games.csv\u001b[39m\u001b[38;5;124m\"\u001b[39m, index\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mFalse\u001b[39;00m)\n\u001b[1;32m     14\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m✅ Saved games.csv\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
      "File \u001b[0;32m/opt/conda/envs/anaconda-2024.02-py310/lib/python3.10/site-packages/pandas/core/frame.py:733\u001b[0m, in \u001b[0;36mDataFrame.__init__\u001b[0;34m(self, data, index, columns, dtype, copy)\u001b[0m\n\u001b[1;32m    727\u001b[0m     mgr \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_init_mgr(\n\u001b[1;32m    728\u001b[0m         data, axes\u001b[38;5;241m=\u001b[39m{\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mindex\u001b[39m\u001b[38;5;124m\"\u001b[39m: index, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcolumns\u001b[39m\u001b[38;5;124m\"\u001b[39m: columns}, dtype\u001b[38;5;241m=\u001b[39mdtype, copy\u001b[38;5;241m=\u001b[39mcopy\n\u001b[1;32m    729\u001b[0m     )\n\u001b[1;32m    731\u001b[0m \u001b[38;5;28;01melif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(data, \u001b[38;5;28mdict\u001b[39m):\n\u001b[1;32m    732\u001b[0m     \u001b[38;5;66;03m# GH#38939 de facto copy defaults to False only in non-dict cases\u001b[39;00m\n\u001b[0;32m--> 733\u001b[0m     mgr \u001b[38;5;241m=\u001b[39m \u001b[43mdict_to_mgr\u001b[49m\u001b[43m(\u001b[49m\u001b[43mdata\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mindex\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mcolumns\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mdtype\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mdtype\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mcopy\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mcopy\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mtyp\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mmanager\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    734\u001b[0m \u001b[38;5;28;01melif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(data, ma\u001b[38;5;241m.\u001b[39mMaskedArray):\n\u001b[1;32m    735\u001b[0m     \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mnumpy\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mma\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m mrecords\n",
      "File \u001b[0;32m/opt/conda/envs/anaconda-2024.02-py310/lib/python3.10/site-packages/pandas/core/internals/construction.py:503\u001b[0m, in \u001b[0;36mdict_to_mgr\u001b[0;34m(data, index, columns, dtype, typ, copy)\u001b[0m\n\u001b[1;32m    499\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m    500\u001b[0m         \u001b[38;5;66;03m# dtype check to exclude e.g. range objects, scalars\u001b[39;00m\n\u001b[1;32m    501\u001b[0m         arrays \u001b[38;5;241m=\u001b[39m [x\u001b[38;5;241m.\u001b[39mcopy() \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mhasattr\u001b[39m(x, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdtype\u001b[39m\u001b[38;5;124m\"\u001b[39m) \u001b[38;5;28;01melse\u001b[39;00m x \u001b[38;5;28;01mfor\u001b[39;00m x \u001b[38;5;129;01min\u001b[39;00m arrays]\n\u001b[0;32m--> 503\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m \u001b[43marrays_to_mgr\u001b[49m\u001b[43m(\u001b[49m\u001b[43marrays\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mcolumns\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mindex\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mdtype\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mdtype\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mtyp\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mtyp\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mconsolidate\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mcopy\u001b[49m\u001b[43m)\u001b[49m\n",
      "File \u001b[0;32m/opt/conda/envs/anaconda-2024.02-py310/lib/python3.10/site-packages/pandas/core/internals/construction.py:114\u001b[0m, in \u001b[0;36marrays_to_mgr\u001b[0;34m(arrays, columns, index, dtype, verify_integrity, typ, consolidate)\u001b[0m\n\u001b[1;32m    111\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m verify_integrity:\n\u001b[1;32m    112\u001b[0m     \u001b[38;5;66;03m# figure out the index, if necessary\u001b[39;00m\n\u001b[1;32m    113\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m index \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m:\n\u001b[0;32m--> 114\u001b[0m         index \u001b[38;5;241m=\u001b[39m \u001b[43m_extract_index\u001b[49m\u001b[43m(\u001b[49m\u001b[43marrays\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    115\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[1;32m    116\u001b[0m         index \u001b[38;5;241m=\u001b[39m ensure_index(index)\n",
      "File \u001b[0;32m/opt/conda/envs/anaconda-2024.02-py310/lib/python3.10/site-packages/pandas/core/internals/construction.py:667\u001b[0m, in \u001b[0;36m_extract_index\u001b[0;34m(data)\u001b[0m\n\u001b[1;32m    664\u001b[0m         \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mPer-column arrays must each be 1-dimensional\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m    666\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m indexes \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m raw_lengths:\n\u001b[0;32m--> 667\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mIf using all scalar values, you must pass an index\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[1;32m    669\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m have_series:\n\u001b[1;32m    670\u001b[0m     index \u001b[38;5;241m=\u001b[39m union_indexes(indexes)\n",
      "\u001b[0;31mValueError\u001b[0m: If using all scalar values, you must pass an index"
     ]
    }
   ],
   "source": [
    "import requests\n",
    "import pandas as pd\n",
    "from datetime import datetime\n",
    "\n",
    "# APIs (replace API_KEY if needed for The Odds API)\n",
    "CFB_API = \"https://api.collegefootballdata.com/games?year=2024&seasonType=regular&week=1\"\n",
    "ODDS_API = f\"https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds?regions=us&markets=h2h&apiKey=YOUR_API_KEY\"\n",
    "\n",
    "def fetch_cfb_games():\n",
    "    r = requests.get(CFB_API)\n",
    "    games = r.json()\n",
    "    df = pd.DataFrame(games)\n",
    "    df.to_csv(\"data/games.csv\", index=False)\n",
    "    print(\"✅ Saved games.csv\")\n",
    "\n",
    "def fetch_odds():\n",
    "    r = requests.get(ODDS_API)\n",
    "    odds = r.json()\n",
    "    df = pd.DataFrame(odds)\n",
    "    df.to_csv(\"data/odds.csv\", index=False)\n",
    "    print(\"✅ Saved odds.csv\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    fetch_cfb_games()\n",
    "    fetch_odds()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "63fea03e-d1df-4b04-abea-3c670bd0bb93",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "anaconda-2024.02-py310",
   "language": "python",
   "name": "conda-env-anaconda-2024.02-py310-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
