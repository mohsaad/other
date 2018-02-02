# Description:
#   A script to say welcome when entering a channel. (In this case, General.)
#
# Notes:
#   This is specifically hardcoded for a person just joining slack. For Muslims of the Bay Slack.


module.exports = (robot) ->
  robot.enter (res) ->
    test_channel = robot.adapter.client.rtm.dataStore.getChannelByName '#general'
    if res.message.room == test_channel.id
        res.send 'Welcome <@' + res.message.user.id + '>! :tada:'
