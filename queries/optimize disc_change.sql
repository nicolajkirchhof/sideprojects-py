CREATE VIEW disc_change AS
SELECT
    t.id AS [track.id],
    t.[name] AS [track.name],
    t.area AS [track.area],
    t.radius_mm AS [track.radius],
    dpa.project_id AS [project.id],
    md.id AS [disc.id],
    md.[name] AS [disc.name],
    md.[type] AS [disc.type],
    md.diameter AS [disc.diameter],
    md.zch_nr AS [disc.zch_nr],
    di.id AS [id],
    wt.id AS [wear_type.id],
    wt.reference_id AS [wear_type.reference_id],
    wt.abbreviation AS [wear_type.abbreviation],
    di.rolling_distance AS [rolling_distance],
    di.cubage AS [cubage],
    di.wear_coefficient AS [wear_coefficient],
    di.tunnelmeter AS [tunnelmeter],
    di.thrust_force,
    di.penetration,
    di.main_drive_speed,
    di.main_drive_torque,
    di.comments AS [comments],
    di.intervention_event_in_id AS [input.id],
    e_from.[id] AS [input.event_id],
    e_from.[from] AS [input.from],
    ie_from.[tunnelmeter] AS [input.position],
    di.wear_in AS [input.wear],
    di.wear_out AS [output.wear],
    di.intervention_event_out_id AS [output.id],
    e_until.[id] AS [output.event_id],
    e_until.[from] AS [output.from],
    ie_until.[tunnelmeter] AS [output.position],
    di.extraordinary as [extraordinary],
    CASE
        WHEN ie_from.id IS NOT NULL THEN ie_from.up_machine_name
        WHEN ie_until.id IS NOT NULL THEN ie_until.up_machine_name
        END AS up_machine_name
--ie_from.[up_machine_name] AS up_machine_name
FROM [disc_input] di
         LEFT JOIN master_disc md on md.id = disc_id
         LEFT JOIN disc_project_assignment dpa on dpa.disc_id = md.id
         LEFT JOIN track t on t.id = track_id
         LEFT JOIN [master_wear_type] wt ON wt.id = wear_type_id
         LEFT JOIN [intervention_event] ie_from ON ie_from.id = di.intervention_event_in_id
         LEFT JOIN [intervention_event] ie_until ON ie_until.id = di.intervention_event_out_id
         LEFT JOIN [event] e_from ON e_from.id = ie_from.event_id
         LEFT JOIN [event] e_until ON e_until.id = ie_until.event_id;
go

