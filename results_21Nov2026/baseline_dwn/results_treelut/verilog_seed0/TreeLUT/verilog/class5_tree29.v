module class5_tree29(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[113]) & i[70]) | ((~i[29]) & i[113] & i[70]) | ((~i[110]) & i[333] & (~i[70])) | (i[289] & (~i[333]) & (~i[70])) | ((~i[289]) & (~i[333]) & (~i[70])) | (i[105] & i[107] & i[29] & i[113] & i[70]) | ((~i[105]) & i[107] & i[29] & i[113] & i[70]) | (i[170] & (~i[107]) & i[29] & i[113] & i[70]) | ((~i[170]) & (~i[107]) & i[29] & i[113] & i[70]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
