import datetime
import pytz
tdatetime = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
tstr = tdatetime.strftime('%Y%m%d_%H%M')
print(tstr)
