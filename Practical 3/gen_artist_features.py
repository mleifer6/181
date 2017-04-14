#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import musicbrainzngs as mbz
import operator
mbz.set_useragent(app = "Harvard CS181 Project", version = 1.0, 
				  contact = "matthewleifer@college.harvard.edu")

"""
count = 0
tags = {}
with open('artists.csv', 'rb') as csvfile:
	rows = csv.reader(csvfile)
	for row in rows:
		# skip header row
		if count == 0:
			count += 1
			continue
		if count >= 10:
			break
		artist_id = row[0]
		print count, artist_id
		try: 
			result = mbz.get_artist_by_id(artist_id, includes = ["tags"])
		except mbz.WebServiceError as exc:
			print("Something went wrong with the request: %s" % exc)
		else:
			#print result
			if 'tag-list' in result['artist']:
				for t in result['artist']['tag-list']:
					tag = t['name']
					if tag in tags:
						tags[tag] += 1
					else:
						tags[tag] = 1
			count += 1

sorted_tags = sorted(tags.items(), key=operator.itemgetter(1))
sorted_tags = sorted_tags[::-1]
"""

# top 400 of 1953 tags
sorted_tags = ["rock", "rock and indie", "american", "british", "classic pop and rock", "uk", "alternative rock", "pop", "pop rock", "english", "electronic", "indie rock", "metal", "usa", "hard rock", "folk", "pop and chart", "hip hop", "punk", "américain", "dance-pop", "dance and electronica", "alternative", "german", "hip hop rnb and dance hall", "indie", "heavy metal", "electropop", "european", "indie pop", "electronica", "progressive rock", "punk rock", "synthpop", "britannique", "jazz", "alternative metal", "singer/songwriter", "england", "folk rock", "new wave", "pop/rock", "blues rock", "hip-hop", "psychedelic rock", "art rock", "contemporary r&b", "swedish", "acoustic rock", "downtempo", "ambient", "canadian", "classical", "soul", "death metal", "dance", "experimental", "post-punk", "pop rap", "pop soul", "composer", "united states", "post-grunge", "80s", "pop punk", "post-hardcore", "blues", "rap", "soundtrack", "soft rock", "90s", "country", "power metal", "rnb", "thrash metal", "metalcore", "power pop", "art pop", "adult contemporary", "french", "00s", "neo-psychedelia", "nu metal", "progressive metal", "alternative dance", "baroque pop", "experimental rock", "black metal", "trip hop", "post-rock", "emo", "soul and reggae", "hardcore punk", "melodic death metal", "funk", "house", "psychedelic pop", "trance", "aor", "norwegian", "hiphop", "us", "britpop", "reggae", "seen live", "finnish", "producer", "rhythm & blues", "scottish", "united kingdom", "folk pop", "gothic metal", "easy listening soundtracks and musicals", "garage rock", "dream pop", "r&b", "ska", "gothic rock", "industrial", "glam rock", "world", "indie folk", "australian", "warp", "10s", "east coast hip hop", "70s", "instrumental", "trip-hop", "chamber pop", "jazz and blues", "post-punk revival", "california", "new metal", "4ad", "glam metal", "piano rock", "symphonic metal", "film soundtrack", "rock & roll", "hardcore", "singer-songwriter", "canada", "electro", "score", "singer", "techno", "rock and roll", "funk rock", "drum and bass", "grunge", "industrial metal", "polish", "conscious hip hop", "hardcore hip hop", "disco", "contemporary folk", "dance-punk", "band", "rap rock", "idm", "symphonic rock", "electro house", "film composer", "sophisti-pop", "groove metal", "country rock", "france", "video game", "jangle pop", "new age", "industrial rock", "n2", "southern rock", "gangsta rap", "indietronica", "français", "classic rock", "latin", "jam band", "female vocalists", "brazilian", "southern hip hop", "vocal jazz", "teen pop", "country pop", "francophone", "film score", "speed metal", "emo-pop", "rapper", "christmas music", "game", "neo-soul", "doom metal", "producteur", "multiple ipi", "garage rock revival", "americana", "vgm", "acoustic", "london", "noise pop", "shoegaze", "melodic hardcore", "sweden", "irish", "progressive pop", "lo-fi indie", "scandinavian", "finland", "male vocalists", "gospel", "classic metal", "rap metal", "synth pop", "gothic", "rapcore", "ebm", "dutch", "blue-eyed soul", "boom bap", "academy award winner", "eurodance", "christian", "electroclash", "california punk", "swing", "opera", "scandinave", "scandinavia", "scandinavie", "noise rock", "ska punk", "space rock", "japanese", "folk metal", "classic thrash metal", "symphonic prog", "lyricist", "ninja tune", "roots rock", "dark ambient", "australia", "pianist", "canadien", "darkwave", "jazz pop", "60s", "electro-industrial", "hip house", "rock opera", "south american", "classic punk", "production music", "jazz fusion", "fixme label mess", "italian", "uk hip hop", "manchester", "los angeles", "new york", "funk metal", "deathcore", "psychedelic", "belgian", "dancehall", "pop reggae", "big band", "rockabilly", "orchestral", "dub", "electric blues", "ambient pop", "modern classical", "compositeur", "español", "chanteur", "christian rock", "spanish", "bebop", "fixme", "west coast hip hop", "german-lyrics", "chillout", "progressive house", "big beat", "skate punk", "political hip hop", "stoner rock", "bossa nova", "soundtracks", "smooth soul", "jazz-rock", "bay area", "a filk artist", "political", "noise", "sunshine pop", "northern irish", "comedy", "math rock", "latin pop", "proto-punk", "icelandic", "ireland", "stoner metal", "pop-rock", "cool jazz", "ambient techno", "art punk", "nwobhm", "hard bop", "vocalist", "sludge metal", "contemporary country", "crossover", "goth", "acid rock", "folktronica", "europop", "standards", "arena rock", "new romantic", "special purpose artist", "psychedelic folk", "parolier", "80s thrash metal", "dubstep", "screamo", "experimental hip hop", "eurovision", "progressive trance", "classic hardcore punk", "college rock", "classic pop punk", "has german audiobooks", "instrumental rock", "austrian", "psychedelic soul", "uk garage", "alternative country", "krautrock", "alternative hip-hop", "television music", "likedis auto", "scotland", "triphop", "glitch", "british blues", "british rhythm & blues", "soundtrack composer", "lo-fi", "mod", "death by gun", "shoegazing", "post-emo", "norway", "bogus artist", "folk-rock", "gratuitous heavy metal umlaut", "trip rock", "instrumental hip hop", "electronic rock", "death by lung cancer", "chamber folk", "parlophone", "berlin", "norsk", "synth funk", "asian", "boogie rock", "irlande", "celtic", "roots reggae", "traditional pop", "piano", "avant-garde jazz", "death by pneumonia", "progressive", "suède", "female vocals", "vocal", "nu-disco", "grime", "alt-country", "oxford", "melodic metalcore", "mpb", "heartland rock", "jamaican", "irlandais", "alternative hip hop", "baggy / madchester", "american thrash metal", "death by overdose", "west coast rock", "piano pop", "electro rock", "surf"]
num_tags = len(sorted_tags)
tag_to_index = {}
for i in range(num_tags):
	tag_to_index[sorted_tags[i]] = i

count = 1
with open('artists.csv', 'rb') as csvfile:
	artist_csv = csv.reader(csvfile)
	# Skip header
	next(artist_csv, None)
	with open("genres_1.csv", 'w') as genres_fh:
		genres_csv = csv.writer(genres_fh, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) 
		genres_csv.writerow(["Id", "Genre"])
		
		for row in artist_csv:
			artist_id = row[0]
			#new_features = [0 for i in range(num_tags)]
			print count, artist_id
			try:
				result = mbz.get_artist_by_id(artist_id, includes = ["tags"])
			except mbz.WebServiceError as exc:
				print("Something went wrong with the request: %s" % exc)
				genres_csv.writerow([artist_id, "Other"])
			else:
				#print result['artist']['tag-list']
				artist_tag = "Other"
				if 'tag-list' in result['artist']:
					counts = {}
					for t in result['artist']['tag-list']:
						if float(t['count']) not in counts:
							counts[float(t['count'])] = []
						counts[float(t['count'])].append(t['name'])

					highest_count = max(counts.keys())
					not_found = True
					for i in range(len(sorted_tags)):
						if artist_tag != "Other":
							not_found = False
							break
						#br = False

						for genre in counts[highest_count]:
							if sorted_tags[i] == genre:
								artist_tag = genre
								#br = True
								#break
						#if br:
						#	break
					if not_found:
						artist_tag = counts[highest_count][0]
				else:
					artist_tag = "Other"

				genres_csv.writerow([artist_id, artist_tag])


				"""
				if 'tag-list' in result['artist']:
					for t in result['artist']['tag-list']:
						tag = t['name']
						if tag in tag_to_index:
							new_features[tag_to_index[tag]] = 1
				"""
				#genres_csv.writerow([artist_id] + new_features)
			count += 1

